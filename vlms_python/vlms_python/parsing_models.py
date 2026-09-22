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
"""VLM models for vlms_python."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from spark_config import Config, register_config

from vlms_python.openai_client.openai_client import OpenAIClient, OpenAIClientConfig
from vlms_python.parsing_output import TaskParsingOutput
from vlms_python.vllm_client.vllm_base import VLLMBase, VLLMBaseConfig


@dataclass
class OpenAIBaseConfig(Config):
    client_config: OpenAIClientConfig = field(default_factory=OpenAIClientConfig)
    system_prompt_path: str = ""
    examples_path: str = ""
    log: bool = False


class OpenAIBase:
    """Base class for OpenAI (Vision) Language Models."""

    def __init__(self, config: OpenAIBaseConfig) -> None:
        """Initialize OpenAI VLM model with given configuration."""
        self.config = config

        system_prompt_path = Path(config.system_prompt_path)
        if system_prompt_path.exists() and system_prompt_path.is_file():
            with system_prompt_path.open() as f:
                self.config.client_config.system_prompt = f.read()

        examples_path = Path(config.examples_path)
        if examples_path.exists() and examples_path.is_file():
            with examples_path.open() as f:
                self.config.client_config.system_prompt += "\n" + f.read()
        self.client = OpenAIClient(self.config.client_config)

    def inference(
        self,
        prompt: Optional[str] = None,
        images: Optional[list[np.ndarray]] = None,
        images_descriptions: Optional[list[str]] = None,
    ):
        """Run inference on the model with given prompt and optional image."
        :param prompt: Text prompt for the model.
        :param images: Optional list of images as NumPy arrays.
        :param images_descriptions: Optional list of image descriptions.
        :return: Model output and success flag.
        """
        result, success = self.client.inference(
            prompt, images, images_descriptions, self.config.log
        )
        return result, success


class OpenAIParsing(OpenAIBase):
    """OpenAI Language Model for task parsing."""

    def __init__(self, config) -> None:
        """Initialize OpenAI VLM model with given configuration."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = OpenAIParsingConfig()
        config.update(kwargs)
        return cls(config)

    def inference(
        self, prompt: Optional[str] = None, image: Optional[list[np.ndarray]] = None
    ) -> TaskParsingOutput:
        """Generate a list of objects of interest, target objects and target rooms.
        :param prompt: Text prompt for the model.
        :param image: Optional image as a NumPy array.
        :return: Model output.
        """
        result, success = super().inference(prompt, image)
        if not success:
            return TaskParsingOutput()
        return TaskParsingOutput(**result, success=success)


class VLLMParsing(VLLMBase):
    """Local vLLM model for task parsing."""

    def __init__(self, config) -> None:
        """Initialize vLLM model with given configuration."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = VLLMParsingConfig()
        config.update(kwargs)
        return cls(config)

    def inference(
        self, prompt: Optional[str] = None, image: Optional[list[np.ndarray]] = None
    ) -> TaskParsingOutput:
        """Generate objects of interest, target objects, and target rooms."""
        result, success = super().inference(prompt, image)
        if not success:
            return TaskParsingOutput()
        return TaskParsingOutput(**result, success=success)


@register_config("task_parsing", name="openai", constructor=OpenAIParsing)
@dataclass
class OpenAIParsingConfig(OpenAIBaseConfig):
    """Configuration for OpenAI VLM model."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


@register_config("task_parsing", name="vllm", constructor=VLLMParsing)
@dataclass
class VLLMParsingConfig(VLLMBaseConfig):
    """Configuration for local vLLM task parsing."""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
