# put images "assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"
python -m gerl.utils.reward_score.jpeg_imcompressibility
python -m gerl.utils.reward_score.multi

# put an image "assets/generated_nyc.jpg" aligned/misaligned with prompt "New York Skyline with "Hello World" written with fireworks on the sky"
python -m gerl.utils.reward_score.ocr

# if use local server
# CUDA_VISIBLE_DEVICES=0 vllm serve ${CHECKPOINT_HOME}/Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --post 9529
export QWEN_VL_OCR_VLLM_URL=http://0.0.0.0:9529/v1
export QWEN_VL_OCR_PATH=${CHECKPOINT_HOME}/Qwen/Qwen2.5-VL-7B-Instruct
python -m gerl.utils.reward_score.vllm
