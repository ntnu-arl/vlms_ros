#!/usr/bin/env python3
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
"""VLM-based room classification node."""

import pathlib
from dataclasses import dataclass
from typing import Any

import rclpy
import spark_config as sc
import yaml
from hydra_msgs.msg import RoomsInfo
from rclpy.node import Node

from vlm_msgs.msg import RoomClassificationOutput

import vlms_ros
from vlms_python import ClassificationOutput, RoomClassificationInput
from vlms_ros import Conversions


@dataclass
class RoomClassificationNodeConfig(sc.Config):
    """Configuration for RoomClassificationNode."""

    model: Any = sc.config_field("room_classification", default="openai")


class RoomClassificationNode(Node):
    """Node that runs room classification."""

    def __init__(self) -> None:
        """Initialize the RoomClassificationNode."""
        super().__init__("room_classification_node")
        # Initialize configuration with YAML and ROS2 parameters
        ros_config_params = (
            self.declare_parameter("config", "").get_parameter_value().string_value
        )
        config_path = (
            self.declare_parameter("config_path", "").get_parameter_value().string_value
        )
        config_path = pathlib.Path(config_path).expanduser().absolute()
        if not config_path.exists() and config_path != "":
            self.get_logger().warn(f"config path '{config_path}' does not exist!")
            self.config = RoomClassificationNodeConfig()
        else:
            self.config = sc.Config.load(RoomClassificationNodeConfig, config_path)
        self.config.update(yaml.safe_load(ros_config_params))

        # Initialize the model
        self._model = self.config.model.create()
        self.get_logger().info(f"Initializing with {self.config.show()}")

        # Publisher and subscriber
        self._room_info_sub = self.create_subscription(
            RoomsInfo,
            "rooms_info",
            self._callback,
            1,
        )
        self._classification_pub = self.create_publisher(
            RoomClassificationOutput,
            "room_classification_output",
            1,
        )
        self.get_logger().info("Finished initializing!")

    def _callback(self, msg: RoomsInfo) -> None:
        """Callback for room info messages.
        :param msg: Incoming RoomsInfo message.
        """
        model_input = [
            RoomClassificationInput(
                room_id=room.room_id,
                images=[Conversions.to_image(keyframe) for keyframe in room.keyframes],
                objects=room.object_labels,
            )
            for room in msg.rooms_info
        ]
        result: ClassificationOutput = self._model.classify(model_input)
        output_msg = RoomClassificationOutput()
        output_msg.header.stamp = self.get_clock().now().to_msg()
        output_msg.classified_rooms = result.labels
        output_msg.confidences = result.confidence
        output_msg.room_ids = result.id
        self._classification_pub.publish(output_msg)


def main() -> None:
    """Start a node."""
    rclpy.init()

    node = None
    try:
        node = RoomClassificationNode()
        vlms_ros.setup_ros_log_forwarding(node)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
