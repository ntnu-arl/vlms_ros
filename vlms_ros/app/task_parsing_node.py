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
"""Node that runs task parsing."""

import pathlib
from dataclasses import dataclass
from typing import Any

import rclpy
import semantic_inference_python
import semantic_inference_python.models as models
import spark_config as sc
import yaml
from hydra_msgs.msg import NewLabels
from rclpy.node import Node

from vlm_msgs.msg import TaskParsingOutput
from vlm_msgs.srv import TaskParsing

import vlms_ros
from vlms_ros.ros_conversions import Conversions


@dataclass
class TaskParsingNodeConfig(sc.Config):
    """Configuration for TaskParsingNode."""

    model: Any = sc.config_field("task_parsing", default="openai")
    embedding_model: Any = sc.config_field("clip", default="open_clip")
    embed_objects: bool = True
    publish_new_labels: bool = False


class TaskParsingNode(Node):
    """Node that runs task parsing."""

    def __init__(self) -> None:
        """Initialize the TaskParsingNode."""
        super().__init__("task_parsing_node")
        ros_config_params = (
            self.declare_parameter("config", "").get_parameter_value().string_value
        )
        config_path = (
            self.declare_parameter("config_path", "").get_parameter_value().string_value
        )
        config_path = pathlib.Path(config_path).expanduser().absolute()
        if not config_path.exists() and config_path != "":
            self.get_logger().warn(f"config path '{config_path}' does not exist!")
            self.config = TaskParsingNodeConfig()
        else:
            self.config = sc.Config.load(TaskParsingNodeConfig, config_path)
        self.config.update(yaml.safe_load(ros_config_params))

        self._model = self.config.model.create()
        self.get_logger().info(f"Initializing with {self.config.show()}")

        device = models.default_device(cuda_device=0)
        self._embedding_model = self.config.embedding_model.create().to(device)
        self._parsing_pub = self.create_publisher(TaskParsingOutput, "output", 1)
        self._new_labels_pub = self.create_publisher(NewLabels, "new_labels", 1)
        self._srv = self.create_service(TaskParsing, "parse", self._handle_task_parsing)
        self.get_logger().info("Finished initializing!")

    def _handle_task_parsing(
        self, request: TaskParsing.Request, response: TaskParsing.Response
    ) -> TaskParsing.Response:
        result = self._model.inference(f"The question is: {request.task}")
        response.output.relevant_objects = result.relevant_objects
        response.output.target_objects = result.target_objects
        response.output.target_rooms = result.target_rooms
        response.output.header.stamp = self.get_clock().now().to_msg()
        if self.config.embed_objects:
            objects_list = result.target_objects
            for obj in result.relevant_objects:
                if obj not in objects_list:
                    objects_list.append(obj)
            embeddings = (
                self._embedding_model.embed_text(objects_list).cpu().numpy().squeeze()
            )
            response.output.objects.names = objects_list
            response.output.objects.features = [
                Conversions.to_feature(x) for x in embeddings
            ]
        self._parsing_pub.publish(response.output)

        if self.config.publish_new_labels:
            new_labels_msg = NewLabels()
            new_labels_msg.header = response.output.header
            new_labels_msg.new_labels = list(
                set(response.output.relevant_objects)
                | set(response.output.target_objects)
            )
            self._new_labels_pub.publish(new_labels_msg)
        return response


def main() -> None:
    """Start a node."""
    rclpy.init()

    node = None
    try:
        node = TaskParsingNode()
        vlms_ros.setup_ros_log_forwarding(node)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
