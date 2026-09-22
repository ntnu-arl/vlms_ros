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
"""Gemini client for vlms_python."""

import json
import os
import re

import numpy as np
from google import genai
from google.genai import types
from PIL import Image

from vlms_python.gemini_client.config import GeminiClientConfig
from vlms_python.misc import LoggerInfo


class GeminiClient:
    """Client for Gemini API to generate vision-language outputs."""

    def __init__(self, config: GeminiClientConfig) -> None:
        """Initialize Gemini client with model and generation settings."""

        self.config = config
        self.api_key = os.getenv(config.api_key_env, "")

        generation_config = {}
        if self.config.max_output_tokens is not None:
            generation_config["max_output_tokens"] = self.config.max_output_tokens
        if self.config.temperature is not None:
            generation_config["temperature"] = self.config.temperature
        if self.config.response_mime_type:
            generation_config["response_mime_type"] = self.config.response_mime_type

        self.generation_config = types.GenerateContentConfig(
            system_instruction=self.config.system_prompt,
            **generation_config,
        )
        self.client = genai.Client(api_key=self.api_key)

    def _to_pil_image(self, image: str | np.ndarray) -> Image.Image:
        """Convert a path or numpy image to a PIL image for Gemini."""
        if isinstance(image, str):
            return Image.open(image)

        if isinstance(image, np.ndarray):
            return Image.fromarray(image.astype(np.uint8))

        raise TypeError("Input must be a file path or a NumPy image array.")

    def _parse_json_from_model_output(self, raw_text: str):
        """Extract and parse JSON from raw model output."""
        if not raw_text:
            raise ValueError("Model output is empty, cannot parse JSON.")

        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.DOTALL
        )

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            match = re.search(r"\{.*\}|\[.*\]", raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                f"Failed to parse JSON from model output: {e}\nRaw text:\n{raw_text}"
            ) from None

    def inference(
        self,
        prompt: str,
        images: list[str] | list[np.ndarray] | None = None,
        images_descriptions: list[str] | None = None,
        log: bool = False,
    ) -> tuple[dict, bool]:
        """Use Gemini's VLM and parse a JSON response."""
        try:
            contents = [prompt or ""]
            if images is not None and len(images) > 0:
                for i, image in enumerate(images):
                    if images_descriptions and i < len(images_descriptions):
                        contents.append(images_descriptions[i])
                    contents.append(self._to_pil_image(image))

            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=self.generation_config,
            )
            raw_text = (getattr(response, "text", "") or "").strip()

            if log:
                LoggerInfo.info(f"[GeminiClient] Prompt: {prompt}")
                LoggerInfo.info(f"[GeminiClient] Response: {raw_text}")

            try:
                parsed = self._parse_json_from_model_output(raw_text)
                return parsed, True
            except Exception as e:
                LoggerInfo.info(
                    f"[GeminiClient] Warning: Failed to parse "
                    f"JSON from model output: {e}"
                )
                return {"error": "Failed to parse JSON"}, False

        except Exception as e:
            LoggerInfo.info(f"[GeminiClient] Error during EQA inference: {e}")
            return {"error": str(e)}, False
