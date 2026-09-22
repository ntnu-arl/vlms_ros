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
"""Llama API client for vlms_python."""

import base64
import json
import os
import re
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from vlms_python.llama_client.config import LlamaClientConfig
from vlms_python.misc import LoggerInfo


class LlamaClient:
    """Client for Llama API vision-language models."""

    def __init__(self, config: LlamaClientConfig) -> None:
        """Initialize Llama API client with model and generation settings."""
        from llama_api_client import LlamaAPIClient

        self.config = config
        self.api_key = os.getenv(config.api_key_env, "")
        self.client = LlamaAPIClient(api_key=self.api_key)

    def _encode_image(self, image: str | np.ndarray) -> str:
        """Convert a path or numpy image to a base64 JPEG string."""
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8))
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

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

    def _message_text(self, message_content: Any) -> str:
        """Normalize Llama API message content to text."""
        if isinstance(message_content, str):
            return message_content

        text = getattr(message_content, "text", None)
        if text is not None:
            return str(text)

        if isinstance(message_content, list):
            parts = []
            for part in message_content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text", part)))
                elif hasattr(part, "text"):
                    parts.append(str(part.text))
                else:
                    parts.append(str(part))
            return "\n".join(parts)

        return str(message_content or "")

    def inference(
        self,
        prompt: str,
        images: list[str] | list[np.ndarray] | None = None,
        images_descriptions: list[str] | None = None,
        log: bool = False,
    ) -> tuple[dict, bool]:
        """Use a Llama API VLM and parse a JSON response."""
        try:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt or ""}]
            if images is not None and len(images) > 0:
                for i, image in enumerate(images):
                    if images_descriptions and i < len(images_descriptions):
                        content.append({"type": "text", "text": images_descriptions[i]})
                    base64_image = self._encode_image(image)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        }
                    )

            request: dict[str, Any] = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": content},
                ],
            }
            if self.config.max_completion_tokens is not None:
                request["max_completion_tokens"] = self.config.max_completion_tokens
            if self.config.temperature is not None:
                request["temperature"] = self.config.temperature
            if self.config.top_p is not None:
                request["top_p"] = self.config.top_p
            if self.config.top_k is not None:
                request["top_k"] = self.config.top_k

            response = self.client.chat.completions.create(**request)
            raw_text = self._message_text(response.completion_message.content).strip()

            if log:
                LoggerInfo.info(f"[LlamaClient] Prompt: {prompt}")
                LoggerInfo.info(f"[LlamaClient] Response: {raw_text}")

            try:
                parsed = self._parse_json_from_model_output(raw_text)
                return parsed, True
            except Exception as e:
                LoggerInfo.info(
                    f"[LlamaClient] Warning: Failed to parse "
                    f"JSON from model output: {e}"
                )
                return {"error": "Failed to parse JSON"}, False

        except Exception as e:
            LoggerInfo.info(f"[LlamaClient] Error during EQA inference: {e}")
            return {"error": str(e)}, False
