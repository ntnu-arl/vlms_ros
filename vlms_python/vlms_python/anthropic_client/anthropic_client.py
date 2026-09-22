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
"""Anthropic Claude client for vlms_python."""

import base64
import json
import os
import re
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from vlms_python.anthropic_client.config import AnthropicClientConfig
from vlms_python.misc import LoggerInfo


class AnthropicClient:
    """Client for Anthropic Claude vision-language models."""

    def __init__(self, config: AnthropicClientConfig) -> None:
        """Initialize Anthropic client with model and generation settings."""
        from anthropic import Anthropic

        self.config = config
        self.api_key = os.getenv(config.api_key_env)

        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = Anthropic()

    def _encode_image(self, image: str | np.ndarray) -> str:
        """Convert a path or numpy image to a base64 JPEG string."""
        if isinstance(image, str):
            with Image.open(image) as pil_img:
                return self._encode_pil_image(pil_img)

        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8))
            return self._encode_pil_image(pil_img)

        raise TypeError("Input must be a file path or a NumPy image array.")

    def _encode_pil_image(self, pil_img: Image.Image) -> str:
        """Encode a PIL image as JPEG base64 for Claude image blocks."""
        buffer = BytesIO()
        pil_img.convert("RGB").save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

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
        """Normalize Anthropic message content blocks to text."""
        if isinstance(message_content, str):
            return message_content

        text = getattr(message_content, "text", None)
        if text is not None:
            return str(text)

        if isinstance(message_content, list):
            parts = []
            for block in message_content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    else:
                        parts.append(str(block))
                elif getattr(block, "text", None) is not None:
                    parts.append(str(block.text))
                else:
                    parts.append(str(block))
            return "\n".join(part for part in parts if part)

        return str(message_content or "")

    def inference(
        self,
        prompt: str,
        images: list[str] | list[np.ndarray] | None = None,
        images_descriptions: list[str] | None = None,
        log: bool = False,
    ) -> tuple[dict, bool]:
        """Use Claude's VLM and parse a JSON response."""
        try:
            content: list[dict[str, Any]] = []
            if images is not None and len(images) > 0:
                for i, image in enumerate(images):
                    base64_image = self._encode_image(image)
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        }
                    )
                    if images_descriptions and i < len(images_descriptions):
                        content.append({"type": "text", "text": images_descriptions[i]})

            content.append({"type": "text", "text": prompt or ""})

            request: dict[str, Any] = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
            if self.config.system_prompt:
                request["system"] = self.config.system_prompt
            if self.config.temperature is not None:
                request["temperature"] = self.config.temperature
            if self.config.top_p is not None:
                request["top_p"] = self.config.top_p
            if self.config.top_k is not None:
                request["top_k"] = self.config.top_k

            response = self.client.messages.create(**request)
            raw_text = self._message_text(response.content).strip()

            if log:
                LoggerInfo.info(f"[AnthropicClient] Prompt: {prompt}")
                LoggerInfo.info(f"[AnthropicClient] Response: {raw_text}")

            try:
                parsed = self._parse_json_from_model_output(raw_text)
                return parsed, True
            except Exception as e:
                LoggerInfo.info(
                    f"[AnthropicClient] Warning: Failed to parse "
                    f"JSON from model output: {e}"
                )
                return {"error": "Failed to parse JSON"}, False

        except Exception as e:
            LoggerInfo.info(f"[AnthropicClient] Error during EQA inference: {e}")
            return {"error": str(e)}, False
