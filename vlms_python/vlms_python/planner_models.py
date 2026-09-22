# BSD 3-Clause License

# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""VLM models for High-Level Planning."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from spark_config import Config, register_config

from vlms_python.parsing_models import OpenAIBase, OpenAIBaseConfig
from vlms_python.vllm_client.vllm_base import VLLMBase, VLLMBaseConfig


class OpenAIHighLevelPlanner(OpenAIBase):
    """High-Level Planner based on OpenAI VLMs."""

    def __init__(self, config: OpenAIBaseConfig) -> None:
        """Initialize the OpenAI High-Level Planner."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "OpenAIHighLevelPlanner":
        """Construct OpenAIHighLevelPlanner from keyword arguments."""
        config = OpenAIHighLevelPlannerConfig()
        config.update(kwargs)
        return cls(config)

    def plan(
        self,
        prompt: Optional[str] = None,
        image: Optional[list[np.ndarray]] = None,
        images_descriptions: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], bool]:
        """Generate a high-level plan based on the given prompt and images.
        :param prompt: Optional text prompt to guide the planning.
        :param image: Optional list of images (as numpy arrays) to provide
                      visual context for the planning.
        :param images_descriptions: Optional list of descriptions for each image.
        :return: A tuple containing a dictionary with the generated plan and
                 any relevant metadata, and a boolean indicating success.
        """
        result, success = super().inference(prompt, image, images_descriptions)
        return result, success


class VLLMHighLevelPlanner(VLLMBase):
    """High-Level Planner based on local OpenAI-compatible vLLM models."""

    def __init__(self, config: VLLMBaseConfig) -> None:
        """Initialize the vLLM High-Level Planner."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "VLLMHighLevelPlanner":
        """Construct VLLMHighLevelPlanner from keyword arguments."""
        config = VLLMHighLevelPlannerConfig()
        config.update(kwargs)
        return cls(config)

    def plan(
        self,
        prompt: Optional[str] = None,
        image: Optional[list[np.ndarray]] = None,
        images_descriptions: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], bool]:
        """Generate a high-level plan based on the given prompt and images."""
        result, success = super().inference(prompt, image, images_descriptions)
        return result, success


@register_config("vlm_planner", name="openai", constructor=OpenAIHighLevelPlanner)
@dataclass
class OpenAIHighLevelPlannerConfig(OpenAIBaseConfig):
    """Configuration for the OpenAI High-Level Planner."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


@register_config("vlm_planner", name="vllm", constructor=VLLMHighLevelPlanner)
@dataclass
class VLLMHighLevelPlannerConfig(VLLMBaseConfig):
    """Configuration for the local vLLM High-Level Planner."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
