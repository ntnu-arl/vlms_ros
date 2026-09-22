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

from vlms_python.anthropic_client.anthropic_base import (
    AnthropicBase,
    AnthropicBaseConfig,
)
from vlms_python.gemini_client.gemini_base import GeminiBase, GeminiBaseConfig
from vlms_python.llama_client.llama_base import LlamaBase, LlamaBaseConfig
from vlms_python.parsing_models import OpenAIBase, OpenAIBaseConfig
from vlms_python.vllm_client.vllm_base import VLLMBase, VLLMBaseConfig


class OpenAIEQA(OpenAIBase):
    """High-Level Planner based on OpenAI VLMs."""

    def __init__(self, config: OpenAIBaseConfig) -> None:
        """Initialize the OpenAI High-Level Planner."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "OpenAIEQA":
        """Construct OpenAIEQA from keyword arguments."""
        config = OpenAIEQAConfig()
        config.update(kwargs)
        return cls(config)

    def answer(
        self,
        prompt: Optional[str] = None,
        image: Optional[list[np.ndarray]] = None,
        images_descriptions: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], bool]:
        """Generate an answer to a question based on the given prompt and images.
        :param prompt: Optional text prompt to guide the answering.
        :param image: Optional list of images (as numpy arrays) to provide
                      visual context for the answering.
        :param images_descriptions: Optional list of descriptions for each image.
        :return: A tuple containing a dictionary with the generated answer and
                 any relevant metadata, and a boolean indicating success.
        """
        result, success = super().inference(prompt, image, images_descriptions)
        return result, success


class GeminiEQA(GeminiBase):
    """EQA planner wrapper for Gemini multimodal models."""

    def __init__(self, config: GeminiBaseConfig) -> None:
        """Initialize the Gemini EQA planner."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "GeminiEQA":
        """Construct GeminiEQA from keyword arguments."""
        config = GeminiEQAConfig()
        config.update(kwargs)
        return cls(config)

    def answer(
        self,
        prompt: Optional[str] = None,
        image: Optional[list[np.ndarray]] = None,
        images_descriptions: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], bool]:
        """Generate an answer to a question using Gemini."""
        result, success = super().inference(prompt, image, images_descriptions)
        return result, success


class LlamaEQA(LlamaBase):
    """EQA planner wrapper for Llama API models."""

    def __init__(self, config: LlamaBaseConfig) -> None:
        """Initialize the Llama EQA planner."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "LlamaEQA":
        """Construct LlamaEQA from keyword arguments."""
        config = LlamaEQAConfig()
        config.update(kwargs)
        return cls(config)

    def answer(
        self,
        prompt: Optional[str] = None,
        image: Optional[list[np.ndarray]] = None,
        images_descriptions: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], bool]:
        """Generate an answer to a question using Llama API."""
        result, success = super().inference(prompt, image, images_descriptions)
        return result, success


class AnthropicEQA(AnthropicBase):
    """EQA planner wrapper for Anthropic Claude multimodal models."""

    def __init__(self, config: AnthropicBaseConfig) -> None:
        """Initialize the Anthropic Claude EQA planner."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "AnthropicEQA":
        """Construct AnthropicEQA from keyword arguments."""
        config = AnthropicEQAConfig()
        config.update(kwargs)
        return cls(config)

    def answer(
        self,
        prompt: Optional[str] = None,
        image: Optional[list[np.ndarray]] = None,
        images_descriptions: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], bool]:
        """Generate an answer to a question using Anthropic Claude."""
        result, success = super().inference(prompt, image, images_descriptions)
        return result, success


class VLLMEQA(VLLMBase):
    """EQA planner wrapper for local OpenAI-compatible vLLM models."""

    def __init__(self, config: VLLMBaseConfig) -> None:
        """Initialize the vLLM EQA planner."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "VLLMEQA":
        """Construct VLLMEQA from keyword arguments."""
        config = VLLMEQAConfig()
        config.update(kwargs)
        return cls(config)

    def answer(
        self,
        prompt: Optional[str] = None,
        image: Optional[list[np.ndarray]] = None,
        images_descriptions: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], bool]:
        """Generate an answer to a question using local vLLM."""
        result, success = super().inference(prompt, image, images_descriptions)
        return result, success


@register_config("eqa_planner", name="openai", constructor=OpenAIEQA)
@dataclass
class OpenAIEQAConfig(OpenAIBaseConfig):
    """Configuration for the OpenAI EQA."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


@register_config("eqa_planner", name="gemini", constructor=GeminiEQA)
@dataclass
class GeminiEQAConfig(GeminiBaseConfig):
    """Configuration for Gemini EQA."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


@register_config("eqa_planner", name="llama", constructor=LlamaEQA)
@dataclass
class LlamaEQAConfig(LlamaBaseConfig):
    """Configuration for Llama EQA."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


@register_config("eqa_planner", name="anthropic", constructor=AnthropicEQA)
@register_config("eqa_planner", name="claude", constructor=AnthropicEQA)
@dataclass
class AnthropicEQAConfig(AnthropicBaseConfig):
    """Configuration for Anthropic Claude EQA."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


@register_config("eqa_planner", name="vllm", constructor=VLLMEQA)
@dataclass
class VLLMEQAConfig(VLLMBaseConfig):
    """Configuration for local vLLM EQA."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
