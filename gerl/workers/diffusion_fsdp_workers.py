# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
The main entry point to run the FlowGRPO algorithm
"""

import logging
import os
import warnings

import psutil
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)

from gerl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_torch_device,
)
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.ray_utils import get_event_loop
from verl.workers.config import FSDPEngineConfig, RolloutConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.fsdp_workers import ActorRolloutRefWorker, get_sharding_strategy
from gerl.workers.config import DiffusersModelConfig
from gerl.workers.rollout import get_rollout_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class DiffusionActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        role="actor",
        enable_activation_offload=False,
        **kwargs,
    ):
        from diffusers import DiffusionPipeline, ModelMixin
        from torch.distributed.fsdp import CPUOffload, MixedPrecision

        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        # TODO (Mike): need refactor structure, this is worker module,
        # we should avoid to import function from specific rollout
        from gerl.workers.rollout.diffusers_rollout.utils import (
            inject_SDE_scheduler_into_pipeline,
        )

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from Diffusers", logger=logger)
        local_path = model_path

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        actor_model_config: dict = DiffusionPipeline.load_config(local_path)
        actor_model_config.update(override_model_config)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        init_context = get_init_weight_context_manager(
            use_meta_tensor=True, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # TODO (Mike): how to pass the actor_model_config into the pipeline?
            pipeline = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                device_map=get_device_name(),
            )
            inject_SDE_scheduler_into_pipeline(
                pipeline, pretrained_model_name_or_path=local_path
            )
            if not hasattr(pipeline, "transformer"):
                raise NotImplementedError(
                    "Only Transformer-based diffusion model is supported now"
                )

            actor_module: ModelMixin = pipeline.transformer

            if use_fused_kernels:
                actor_module.fuse_qkv_projections()

            if enable_gradient_checkpointing:
                actor_module.enable_gradient_checkpointing()

            if self._is_lora:
                print("Applying LoRA to actor module")

                lora_adapter_path = self.config.model.get("lora_adapter_path")
                if lora_adapter_path is not None:
                    from peft import PeftModel

                    print(
                        f"Loading pre-trained LoRA adapter to {role} from: {lora_adapter_path}"
                    )

                    # Copy adapter to local if needed
                    local_adapter_path = copy_to_local(
                        lora_adapter_path,
                        use_shm=self.config.model.get("use_shm", False),
                    )

                    actor_module = PeftModel.from_pretrained(
                        actor_module, local_adapter_path, is_trainable=True
                    )
                else:
                    # Convert config to regular Python types before creating PEFT model
                    lora_config = {
                        "r": self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        # TODO (Mike): make init_lora_weights configurable
                        "init_lora_weights": "gaussian",
                        "target_modules": convert_to_regular_types(
                            self.config.model.target_modules
                        ),
                        "exclude_modules": convert_to_regular_types(
                            self.config.model.exclude_modules
                        ),
                        # TODO (Mike): double check default bias value
                        "bias": "none",
                    }
                    actor_module = get_peft_model(
                        actor_module, LoraConfig(**lora_config)
                    )

        self.use_orig_params = fsdp_config.get("use_orig_params", False)

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, (
                "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            )
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=True,
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = (
                    None if role == "actor" else CPUOffloadPolicy(pin_memory=True)
                )

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(
                    fsdp_size=self.device_mesh.shape[-1]
                ),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(
                actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing
            )

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import (
                get_constant_schedule_with_warmup,
                get_cosine_schedule_with_warmup,
            )

            actor_optimizer = build_optimizer(
                actor_module_fsdp.parameters(), optim_config
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(
                    f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}"
                )

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(
                    f"LR scheduler type {lr_scheduler_type} is not supported"
                )

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return (
            pipeline,
            actor_module_fsdp,
            actor_optimizer,
            actor_lr_scheduler,
            actor_model_config,
        )

    def _build_rollout(self, actor_rollout_module):
        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: DiffusersModelConfig = omega_conf_to_dataclass(
            self.config.model, dataclass_type=DiffusersModelConfig
        )
        self.model_config = model_config

        # 2. build rollout device mesh
        infer_tp = (
            self.config.rollout.tensor_model_parallel_size
            * self.config.rollout.data_parallel_size
        )
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name,
            mesh_shape=(dp, infer_tp, infer_pp),
            mesh_dim_names=["dp", "infer_tp", "infer_pp"],
        )
        rollout_name = self.config.rollout.name

        if rollout_name == "hf":
            self._register_dispatch_collect_info(
                "rollout", dp_rank=self.rank, is_collect=True
            )
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout",
                dp_rank=rollout_device_mesh["dp"].get_local_rank(),
                is_collect=is_collect,
            )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(
            gen_dp_rank + 1000
        )  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model
        log_gpu_memory_usage(
            f"Before building {self.config.rollout.name} rollout", logger=logger
        )
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            rollout_module=actor_rollout_module,
            config=rollout_config,
            model_config=model_config,
            device_mesh=rollout_device_mesh,
        )
        log_gpu_memory_usage(
            f"After building {self.config.rollout.name} rollout", logger=logger
        )

        # Full params
        if (
            torch.distributed.get_world_size() == 1
            and fsdp_version(self.actor_module_fsdp) == 1
        ):
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # used for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For sync mode, we directly switch to trainer mode here.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        if rollout_config.mode == "sync" and self._is_actor:
            loop = get_event_loop()
            loop.run_until_complete(self.trainer_mode())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from gerl.workers.actor import DiffusersPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_rollout_module,
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get(
                    "enable_gradient_checkpointing", False
                ),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get(
                    "enable_activation_offload", False
                ),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage(
                    "After offload actor model during init", logger=logger
                )

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage(
                    "After offload actor optimizer during init", logger=logger
                )

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DiffusersPPOActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                pipeline=self.actor_rollout_module,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_rollout:
            self._build_rollout(self.actor_rollout_module)

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DiffusersPPOActor(
                config=self.config.ref,
                actor_module=self.ref_module_fsdp,
                pipeline=self.actor_rollout_module,
            )

        if self._is_actor:
            # TODO: support flopscounter
            # self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                checkpoint_config=self.config.actor.checkpoint,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.actor_optimizer, device_id=get_device_id()
            )

        data = data.to(
            "cpu"
        )  # data will to device with each micro batch on actor.update_policy

        metrics = self.actor.update_policy(data=data)
        metrics["perf/max_memory_allocated_gb"] = (
            get_torch_device().max_memory_allocated() / (1024**3)
        )
        metrics["perf/max_memory_reserved_gb"] = (
            get_torch_device().max_memory_reserved() / (1024**3)
        )
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
        self.actor_lr_scheduler.step()

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})

        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage(
                "After offload actor model during update_actor", logger=logger
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage(
                "After offload actor optimizer during update_actor", logger=logger
            )

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts):
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = (
            topk_reduce_ratio_min_max(timing_generate["generate_sequences"])
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = (
            self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        )
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = (
            self.config.rollout.log_prob_micro_batch_size_per_gpu
        )
        # perform recompute log_prob
        with adapter_ctx:
            output = self.actor.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={"old_log_probs": output})

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage(
                "After offload actor model during compute_log_prob", logger=logger
            )

        return output


class AsyncDiffusionActorRolloutRefWorker(DiffusionActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        raise NotImplementedError
