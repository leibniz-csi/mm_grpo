# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import asyncio
import base64
import logging
import os
import statistics
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Union

import instructor
import numpy as np
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from PIL import Image
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)

from .scorer import Scorer

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


# Default Pydantic schema for structured output
class DefaultScoreResponse(BaseModel):
    """Default structured response for image evaluation."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score from 0.0 to 1.0, where 1.0 is perfect and 0.0 is completely wrong.",
    )
    reasoning: str = Field(
        description="Brief explanation of the score.",
    )


class OpenAIClientScorer(Scorer):
    """
    A flexible OpenAI-compatible API client scorer that supports:
    - Multiple providers (OpenAI, OpenRouter, local VLLM, LM Studio, etc.)
    - Structured output via Pydantic schemas with instructor
    - Variance reduction through multiple samples and aggregation
    - Customizable evaluation and system prompts
    - Automatic retry with exponential backoff using tenacity

    Args:
        base_url (str, optional): The base URL for the API endpoint.
            Can also be set via OPENAI_CLIENT_BASE_URL env var or .env file.
        api_key (str, optional): API key for authentication.
            Can also be set via OPENAI_API_KEY env var or .env file.
        model (str, optional): Model name to use. Defaults to "gpt-4o-mini".
            Can also be set via OPENAI_CLIENT_MODEL env var or .env file.
        system_prompt (str, optional): System prompt for the assistant.
            Defaults to a helpful image evaluation assistant message.
        evaluation_prompt (str, optional): Custom evaluation prompt template.
            Should include {prompt} placeholder if using image prompts.
        response_schema (BaseModel, optional): Pydantic model for structured outputs.
            Defaults to DefaultScoreResponse with score and reasoning fields.
        score_key (str): Key in the response schema that contains the score.
            Defaults to "score".
        k_samples (int): Number of samples to generate for variance reduction. Defaults to 1.
        aggregation_method (str): Method to aggregate multiple samples.
            Options: "mean", "median". Defaults to "mean".
        max_tokens (int): Maximum tokens in response. Defaults to 512.
        temperature (float): Sampling temperature. Defaults to 0.7.
        max_retries (int): Maximum number of retry attempts. Defaults to 3.
        retry_min_wait (float): Minimum wait time between retries in seconds. Defaults to 2.
        retry_max_wait (float): Maximum wait time between retries in seconds. Defaults to 20.
        api_timeout (float): API request timeout in seconds. Defaults to 180.0.
        extra_body (dict, optional): Extra parameters to pass in request body (e.g., for OpenRouter).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        evaluation_prompt: Optional[str] = None,
        response_schema: Optional[type[BaseModel]] = None,
        score_key: str = "score",
        k_samples: int = 1,
        aggregation_method: Literal["mean", "median"] = "mean",
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_min_wait: float = 2.0,
        retry_max_wait: float = 20.0,
        api_timeout: float = 180.0,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Get configuration from environment or parameters
        self.base_url = base_url or os.environ.get("OPENAI_CLIENT_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.environ.get("OPENAI_CLIENT_MODEL", "gpt-4o-mini")

        # Prompt settings
        self.system_prompt = system_prompt or (
            "You are a helpful image evaluation assistant. "
            "Provide accurate and objective assessments of image quality."
        )
        self.evaluation_prompt = evaluation_prompt or (
            "Evaluate the quality of this image based on the prompt: {prompt}\n"
            "Provide a score between 0.0 and 1.0, where 1.0 is perfect and 0.0 is completely wrong."
        )

        # Structured output settings
        self.response_schema = response_schema or DefaultScoreResponse
        self.score_key = score_key

        # Variance reduction settings
        self.k_samples = max(1, k_samples)
        self.aggregation_method = aggregation_method

        # Model parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = extra_body or {}

        # Retry settings
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.api_timeout = api_timeout

        # Initialize async client and event loop
        self._loop = asyncio.new_event_loop()
        self._instructor_client = None

    def _get_client(self):
        """Get or create instructor-wrapped async OpenAI client."""
        if self._instructor_client is None:
            base_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.api_timeout,
                max_retries=0,  # We handle retries with tenacity
            )
            self._instructor_client = instructor.from_openai(
                base_client, mode=instructor.Mode.JSON
            )

        return self._instructor_client

    async def _async_query_with_retry(
        self,
        messages: List[Dict[str, Any]],
    ) -> Any:
        """Query OpenAI API with automatic retry using tenacity."""

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=1, min=self.retry_min_wait, max=self.retry_max_wait
            ),
            retry=retry_if_exception_type(
                (APITimeoutError, APIError, RateLimitError, asyncio.TimeoutError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.DEBUG),
        )
        async def _query():
            client = self._get_client()
            # Use instructor for structured generation
            return await client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=self.response_schema,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body=self.extra_body,
            )

        return await _query()

    async def _async_query_with_k_samples(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Any]:
        """Query API k times for variance reduction."""
        results = await asyncio.gather(
            *(self._async_query_with_retry(messages) for _ in range(self.k_samples)),
            return_exceptions=True,
        )
        return results

    async def _async_process_batch(
        self,
        images_base64: List[str],
        prompts: Optional[List[str]] = None,
    ) -> List[List[Any]]:
        """Process a batch of images, getting k samples for each."""
        messages_batch = [
            self._prepare_messages(img_b64, prompt)
            for img_b64, prompt in zip(
                images_base64, prompts if prompts else [None] * len(images_base64)
            )
        ]

        results = await asyncio.gather(
            *(self._async_query_with_k_samples(messages) for messages in messages_batch)
        )
        return results

    def _prepare_messages(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Prepare messages for the API call."""
        # Format the evaluation prompt
        eval_text = self.evaluation_prompt
        if prompt and "{prompt}" in eval_text:
            eval_text = eval_text.format(prompt=prompt)

        content = [
            {
                "type": "image_url",
                "image_url": {"url": image_base64},
            },
            {
                "type": "text",
                "text": eval_text,
            },
        ]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

        return messages

    @staticmethod
    def _pil_image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded_image}"

    def _extract_score(self, response: Any) -> float:
        """Extract numerical score from response.

        Args:
            response: Pydantic model instance from instructor

        Returns:
            float: Score value between 0.0 and 1.0
        """
        try:
            # Response is a Pydantic model from instructor
            if isinstance(response, BaseModel):
                score = getattr(response, self.score_key, None)
                if score is not None:
                    return float(score)
                logger.warning(
                    f"Score key '{self.score_key}' not found in response: {response}"
                )
                return 0.0

            logger.warning(f"Unexpected response type: {type(response)}")
            return 0.0

        except Exception as e:
            logger.error(f"Error extracting score from response: {e}")
            return 0.0

    def _aggregate_scores(self, scores: List[float]) -> float:
        """Aggregate multiple scores using the configured method.

        Args:
            scores: List of scores to aggregate

        Returns:
            float: Aggregated score
        """
        if not scores:
            return 0.0
        if len(scores) == 1:
            return scores[0]

        if self.aggregation_method == "mean":
            return statistics.mean(scores)
        elif self.aggregation_method == "median":
            return statistics.median(scores)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, torch.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Calculate reward scores for images using OpenAI-compatible API.

        Args:
            images: List of PIL Images, or numpy/torch tensors in NCHW format
            prompts: Optional list of text prompts corresponding to each image

        Returns:
            List of reward scores (floats between 0 and 1)
        """
        # Convert tensors to PIL images if needed
        if isinstance(images, (np.ndarray, torch.Tensor)):
            if images.ndim == 3:
                images = (
                    images.unsqueeze(0)
                    if isinstance(images, torch.Tensor)
                    else np.expand_dims(images, 0)
                )
            images = self.array_to_images(images)

        # Convert images to base64
        images_base64 = [self._pil_image_to_base64(img) for img in images]

        # Get k samples for each image
        results_batch = self._loop.run_until_complete(
            self._async_process_batch(images_base64, prompts)
        )

        logger.debug(f"API responses (count): {len(results_batch)}")

        # Extract and aggregate scores for each image
        final_scores = []
        for k_responses in results_batch:
            # Filter out exceptions and extract scores
            scores = []
            for i, response in enumerate(k_responses):
                if isinstance(response, Exception):
                    logger.warning(
                        f"Sample {i + 1}/{self.k_samples} failed: {response}"
                    )
                else:
                    score = self._extract_score(response)
                    scores.append(score)

            # If all failed, return low score
            if not scores:
                logger.error(f"All {self.k_samples} evaluations failed for an image")
                final_scores.append(0.1)
            else:
                # Aggregate the successful scores
                aggregated_score = self._aggregate_scores(scores)
                final_scores.append(aggregated_score)
                logger.debug(
                    f"Scores: {scores}, Aggregated ({self.aggregation_method}): {aggregated_score:.3f}"
                )

        return final_scores


if __name__ == "__main__":
    """Test the OpenAI client scorer."""
    print("=" * 60)
    print("Testing OpenAI Client Scorer")
    print("=" * 60)

    # Create scorer
    scorer = OpenAIClientScorer(
        k_samples=2,
        aggregation_method="mean",
        temperature=0.7,
    )

    print("\n✅ Scorer created")
    print(f"  Base URL: {scorer.base_url}")
    print(f"  Model: {scorer.model}")
    print(f"  K samples: {scorer.k_samples}")
    print(f"  Aggregation: {scorer.aggregation_method}")

    # Create test images
    test_images = []
    for color in [(255, 0, 0), (0, 255, 0)]:
        img = Image.new("RGB", (128, 128), color=color)
        test_images.append(img)

    prompts = ["A red square", "A green square"]

    print(f"\n✅ Evaluating {len(test_images)} images...")
    scores = scorer(images=test_images, prompts=prompts)

    print("\nResults:")
    for i, (prompt, score) in enumerate(zip(prompts, scores)):
        print(f"  {i + 1}. {prompt}: {score:.3f}")

    print("\n✅ Test completed!")
