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
"""Base model wrapper for Gemini VLMs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from spark_config import Config

from vlms_python.gemini_client.config import GeminiClientConfig
from vlms_python.gemini_client.gemini_client import GeminiClient


@dataclass
class GeminiBaseConfig(Config):
    """Configuration for Gemini VLM wrappers."""

    client_config: GeminiClientConfig = field(default_factory=GeminiClientConfig)
    system_prompt_path: str = ""
    examples_path: str = ""
    log: bool = False


class GeminiBase:
    """Base class for Gemini vision-language models."""

    def __init__(self, config: GeminiBaseConfig) -> None:
        """Initialize Gemini VLM model with given configuration."""
        self.config = config

        system_prompt_path = Path(config.system_prompt_path)
        if system_prompt_path.exists() and system_prompt_path.is_file():
            with system_prompt_path.open() as f:
                self.config.client_config.system_prompt = f.read()

        examples_path = Path(config.examples_path)
        if examples_path.exists() and examples_path.is_file():
            with examples_path.open() as f:
                self.config.client_config.system_prompt += "\n" + f.read()

        self.client = GeminiClient(self.config.client_config)

    def inference(
        self,
        prompt: Optional[str] = None,
        images: Optional[list[np.ndarray]] = None,
        images_descriptions: Optional[list[str]] = None,
    ):
        """Run inference on Gemini with given prompt and optional images."""
        result, success = self.client.inference(
            prompt, images, images_descriptions, self.config.log
        )
        return result, success
