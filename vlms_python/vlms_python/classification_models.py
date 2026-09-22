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
"""VLM-based classification models."""

from dataclasses import dataclass

from spark_config import Config, register_config

from vlms_python.classification_io import ClassificationOutput, RoomClassificationInput
from vlms_python.misc import LoggerError
from vlms_python.parsing_models import OpenAIBase, OpenAIBaseConfig
from vlms_python.vllm_client.vllm_base import VLLMBase, VLLMBaseConfig


class OpenAIRoomClassification(OpenAIBase):
    """OpenAI-based room classification model."""

    def __init__(self, config) -> None:
        """Initialize OpenAI room classification model with given configuration.
        :param config: Configuration
        """
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "OpenAIRoomClassification":
        """Construct OpenAIRoomClassification from keyword arguments.
        :param kwargs: Keyword arguments for configuration.
        :return: Instance of OpenAIRoomClassification.
        """
        config = RoomClassificationModelConfig()
        config.update(kwargs)
        return cls(config)

    def classify(
        self, model_input: list[RoomClassificationInput], room_labels: list[str] = None
    ) -> ClassificationOutput:
        """Classify rooms based on input data.
        :param model_input: List of RoomClassificationInput instances.
        :param room_labels: List of valid room labels.
        :return: ClassificationOutput instance with classified rooms.
        """
        # Prepare prompts and images for each room
        output = ClassificationOutput()
        for room in model_input:
            prompt = f"Objects in the room: {', '.join(room.objects)}."
            if room_labels:
                prompt += f" Possible room types: {', '.join(room_labels)}."
            # Run inference using the base class method
            result, success = self.inference(prompt, room.images)
            if not success:
                LoggerError.error(
                    f"OpenAI inference failed during classification"
                    f"of room {room.room_id}."
                )
                return output

            output.labels.append(result["label"])
            output.confidence.append(result["confidence"])
            output.id.append(room.room_id)

        return output


class VLLMRoomClassification(VLLMBase):
    """vLLM-based room classification model."""

    def __init__(self, config) -> None:
        """Initialize vLLM room classification model with given configuration."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs) -> "VLLMRoomClassification":
        """Construct VLLMRoomClassification from keyword arguments."""
        config = VLLMRoomClassificationConfig()
        config.update(kwargs)
        return cls(config)

    def classify(
        self, model_input: list[RoomClassificationInput], room_labels: list[str] = None
    ) -> ClassificationOutput:
        """Classify rooms based on input data."""
        output = ClassificationOutput()
        for room in model_input:
            prompt = f"Objects in the room: {', '.join(room.objects)}."
            if room_labels:
                prompt += f" Possible room types: {', '.join(room_labels)}."

            result, success = self.inference(prompt, room.images)
            if not success:
                LoggerError.error(
                    f"vLLM inference failed during classification "
                    f"of room {room.room_id}."
                )
                return output

            output.labels.append(result["label"])
            output.confidence.append(result["confidence"])
            output.id.append(room.room_id)

        return output


@register_config(
    "room_classification", name="openai", constructor=OpenAIRoomClassification
)
@dataclass
class RoomClassificationModelConfig(OpenAIBaseConfig):
    """Configuration for OpenAI room classification model."""

    @classmethod
    def load(cls, filepath) -> "RoomClassificationModelConfig":
        """Load configuration from a file.
        :param filepath: Path to the configuration file.
        :return: Instance of RoomClassificationModelConfig.
        """
        return Config.load(cls, filepath)


@register_config("room_classification", name="vllm", constructor=VLLMRoomClassification)
@dataclass
class VLLMRoomClassificationConfig(VLLMBaseConfig):
    """Configuration for local vLLM room classification model."""

    @classmethod
    def load(cls, filepath) -> "VLLMRoomClassificationConfig":
        """Load configuration from a file."""
        return Config.load(cls, filepath)
