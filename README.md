# VLMs ROS2

A ROS2 suite for Vision Language Model (VLM) inference across different tasks. This package provides ROS2 nodes and services for leveraging VLMs in robotics applications, with a focus on semantic understanding and task parsing.

## Description

This repository provides a modular framework for integrating Vision Language Models into ROS2-based robotic systems. The package is designed with extensibility in mind, allowing for easy integration of different VLM backends (API-based and local models) while maintaining a consistent interface for ROS2 applications.

The package consists of three main components:

- **`vlm_msgs`**: ROS2 message and service definitions for VLM-related communication
- **`vlms_python`**: Core Python library containing VLM model implementations and client interfaces
- **`vlms_ros`**: ROS2 nodes and launch files for running VLM inference tasks

## Currently Implemented

### Supported VLM Backends
- **OpenAI API**: Integration with OpenAI's vision language models (e.g., GPT-4o) via their API
- **Google Gemini API**: Integration with Gemini vision models via the `google-genai` SDK for EQA planning
- **Meta Llama API**: Integration with Llama vision models via the official `llama-api-client` SDK for EQA planning
- **Anthropic API**: Claude multimodal models for EQA planning
- **OpenAI-compatible vLLM**: Locally or remotely hosted models for all implemented tasks

Backend support is task-specific:

| Task | OpenAI | Gemini | Llama API | Anthropic | vLLM |
| --- | --- | --- | --- | --- | --- |
| Room Classification | Yes | No | No | No | Yes |
| Task Parsing | Yes | No | No | No | Yes |
| EQA Planning | Yes | Yes | Yes | Yes | Yes |

The EQA planner model wrappers live in `vlms_python` and are consumed by downstream planner configurations, such as `hvlm_planner`.

### Implemented Tasks

#### 1. Room Classification
Classifies rooms based on visual information and detected objects.

- **Node**: `room_classification_node`
- **Input**: Subscribes to `rooms_info` topic (from `hydra_msgs/RoomsInfo`)
- **Output**: Publishes `room_classification_output` topic (`vlm_msgs/RoomClassificationOutput`)
  - Classified room labels
  - Confidence scores
  - Room IDs

#### 2. Task Parsing
Parses natural language task descriptions to extract semantic information.

- **Node**: `task_parsing_node`
- **Service**: `parse` service (`vlm_msgs/TaskParsing`)
- **Outputs**:
  - `output` topic (`vlm_msgs/TaskParsingOutput`): Contains target objects, target rooms, and relevant objects
  - `new_labels` topic (`hydra_msgs/NewLabels`): Publishes new object labels extracted from the task

The `hflex_eqa` branch unifies the former desktop and Jetson Thor variants. The same code supports API backends and OpenAI-compatible hosted/local vLLM servers; no deployment-specific branch switch is required.

## Installation

### Prerequisites

- ROS2 (tested with ROS2 Jazzy on Ubuntu 24.04, but should work with other distributions)
- Python 3.12
- API key for the provider you use:
  - `OPENAI_API_KEY` for OpenAI
  - `GEMINI_API_KEY` for Gemini
  - `LLAMA_API_KEY` for Meta Llama API
  - `ANTHROPIC_API_KEY` for Anthropic

### Building the Package

1. Clone this repository into your ROS2 workspace:
```bash
cd ~/your_ros2_ws/src
git clone git@github.com:ntnu-arl/vlms_ros.git vlms_ros
```

2. Install Python dependencies:
```bash
cd vlms_ros/vlms_python
pip install -r requirements.txt
```

3. Build the workspace:
```bash
cd ~/your_ros2_ws
colcon build --symlink-install
source install/local_setup.bash
```

### Configuration

1. Set the API key for the backend you are using:
```bash
# Export only the keys for the providers you plan to use.
export OPENAI_API_KEY="your-api-key-here"
export GEMINI_API_KEY="your-api-key-here"
export LLAMA_API_KEY="your-api-key-here"
export ANTHROPIC_API_KEY="your-api-key-here"
# Local vLLM endpoints do not require a real key by default.
```

2. Configure the models by editing the YAML configuration files:
   - Room Classification: [config.yaml](./vlms_ros/config/room_classification/config.yaml)
   - Task Parsing: [config.yaml](./vlms_ros/config/task_parsing/config.yaml)

Example configuration:
```yaml
model:
  type: openai
  log: true
  client_config:
    model: gpt-4o
```

EQA planner backends are registered under the `eqa_planner` config group. Example VLM selections:

```yaml
vlm:
  type: openai
  client_config:
    model: gpt-4o
  log: true
```

```yaml
vlm:
  type: gemini
  client_config:
    model: gemini-2.5-flash
    api_key_env: GEMINI_API_KEY
    response_mime_type: application/json
  log: true
```

```yaml
vlm:
  type: llama
  client_config:
    model: Llama-4-Maverick-17B-128E-Instruct-FP8
    api_key_env: LLAMA_API_KEY
  log: true
```

```yaml
vlm:
  type: anthropic
  client_config:
    model: claude-sonnet-4-5
    api_key_env: ANTHROPIC_API_KEY
  log: true
```

```yaml
vlm:
  type: vllm
  client_config:
    model: RedHatAI/gemma-3-12b-it-quantized.w4a16
    base_url: http://127.0.0.1:8000/v1
    api_key: not-needed
    use_system_message: false
  log: true
```

3. Customize system prompts and examples:
   - Room Classification: [system_prompt.txt](./vlms_ros/config/room_classification/system_prompt.txt)
   - Task Parsing: [system_prompt.txt](./vlms_ros/config/task_parsing/system_prompt.txt) and [examples.txt](./vlms_ros/config/task_parsing/examples.txt)

## Usage Examples

### Running Room Classification Node

Launch the room classification node:
```bash
ros2 launch vlms_ros room_classification.launch.yaml
```

Or with custom configuration:
```bash
ros2 launch vlms_ros room_classification.launch.yaml \
  config_path:=/path/to/your/config.yaml \
  system_prompt_path:=/path/to/your/system_prompt.txt
```

The node will:
- Subscribe to `rooms_info` topic
- Process room information and classify rooms using the VLM
- Publish results to `room_classification_output` topic

**Example: Viewing classification results**
```bash
ros2 topic echo /vlms/room_classification_output
```

### Running Task Parsing Node

Launch the task parsing node:
```bash
ros2 launch vlms_ros task_parsing.launch.yaml
```

Or with custom configuration:
```bash
ros2 launch vlms_ros task_parsing.launch.yaml \
  config_path:=/path/to/your/config.yaml \
  system_prompt_path:=/path/to/your/system_prompt.txt \
  examples_path:=/path/to/your/examples.txt
```

**Example: Parsing a task via service call**
```bash
ros2 service call /vlms/parse vlm_msgs/TaskParsing "{task: 'Bring me a cup from the kitchen'}"
```

**Example: Viewing parsing results**
```bash
ros2 topic echo /vlms/output
ros2 topic echo /vlms/new_labels
```

## Package Structure

```
vlms_ros/
├── vlm_msgs/                    # ROS2 message definitions
│   ├── msg/
│   │   ├── RoomClassificationOutput.msg
│   │   └── TaskParsingOutput.msg
│   └── srv/
│       └── TaskParsing.srv
├── vlms_python/                 # Core Python library
│   ├── vlms_python/
│   │   ├── classification_models.py
│   │   ├── classification_io.py
│   │   ├── eqa_models.py
│   │   ├── parsing_models.py
│   │   ├── parsing_output.py
│   │   ├── gemini_client/       # Gemini API client
│   │   ├── llama_client/        # Meta Llama API client
│   │   ├── anthropic_client/    # Anthropic API client
│   │   ├── vllm_client/         # OpenAI-compatible local/hosted client
│   │   └── openai_client/       # OpenAI API client
│   └── requirements.txt
└── vlms_ros/                    # ROS2 nodes and launch files
    ├── app/
    │   ├── room_classification_node.py
    │   └── task_parsing_node.py
    ├── config/
    │   ├── room_classification/
    │   └── task_parsing/
    └── launch/
        ├── room_classification.launch.yaml
        └── task_parsing.launch.yaml
```

## License

BSD 3-Clause License

Copyright (c) 2026, NTNU Autonomous Robots Lab

## Maintainer

Albert Gassol Puigjaner (albert.g.puigjaner@ntnu.no)
