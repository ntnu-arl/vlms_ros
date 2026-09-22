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
"""OpenAI-compatible local vLLM client for vlms_python."""

import base64
import json
import re
from io import BytesIO
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image

from vlms_python.misc import LoggerInfo
from vlms_python.vllm_client.config import VLLMClientConfig


class VLLMClient:
    """Client for local vLLM servers exposing the OpenAI chat API."""

    def __init__(self, config: VLLMClientConfig) -> None:
        """Initialize vLLM client with model, base URL, and generation settings."""
        self.config = config
        client_kwargs: dict[str, Any] = {
            "base_url": self.config.base_url,
            "api_key": self.config.api_key,
        }
        if self.config.timeout is not None:
            client_kwargs["timeout"] = self.config.timeout
        self.client = OpenAI(**client_kwargs)

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
        """Normalize OpenAI-compatible chat message content to text."""
        if isinstance(message_content, str):
            return message_content

        if isinstance(message_content, list):
            parts = []
            for part in message_content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
                elif hasattr(part, "text"):
                    parts.append(str(part.text))
                else:
                    parts.append(str(part))
            return "\n".join(part for part in parts if part)

        return str(message_content or "")

    def _make_messages(
        self,
        prompt: str,
        images: list[str] | list[np.ndarray] | None,
        images_descriptions: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Build chat-completions messages for vLLM multimodal models."""
        user_text = prompt or ""
        messages: list[dict[str, Any]] = []
        if self.config.use_system_message:
            messages.append({"role": "system", "content": self.config.system_prompt})
        elif self.config.system_prompt:
            user_text = f"{self.config.system_prompt}\n{user_text}"

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if images is not None and len(images) > 0:
            for i, image in enumerate(images):
                base64_image = self._encode_image(image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    }
                )
                if images_descriptions and i < len(images_descriptions):
                    content[0]["text"] += (
                        f"\n\n[Caption of image {i + 1}: {images_descriptions[i]}]"
                    )

        messages.append({"role": "user", "content": content})
        return messages

    def inference(
        self,
        prompt: str,
        images: list[str] | list[np.ndarray] | None = None,
        images_descriptions: list[str] | None = None,
        log: bool = False,
    ) -> tuple[dict, bool]:
        """Use a local vLLM VLM and parse a JSON response."""
        try:
            request: dict[str, Any] = {
                "model": self.config.model,
                "messages": self._make_messages(prompt, images, images_descriptions),
            }
            if self.config.max_tokens is not None:
                request["max_tokens"] = self.config.max_tokens
            if self.config.temperature is not None:
                request["temperature"] = self.config.temperature
            if self.config.top_p is not None:
                request["top_p"] = self.config.top_p
            if self.config.extra_body:
                request["extra_body"] = self.config.extra_body

            response = self.client.chat.completions.create(**request)
            raw_text = self._message_text(response.choices[0].message.content).strip()

            if log:
                LoggerInfo.info(f"[VLLMClient] Prompt: {prompt}")
                LoggerInfo.info(f"[VLLMClient] Response: {raw_text}")

            try:
                parsed = self._parse_json_from_model_output(raw_text)
                return parsed, True
            except Exception as e:
                LoggerInfo.info(
                    f"[VLLMClient] Warning: Failed to parse JSON from model output: {e}"
                )
                return {"error": "Failed to parse JSON"}, False

        except Exception as e:
            LoggerInfo.info(f"[VLLMClient] Error during inference: {e}")
            return {"error": str(e)}, False
