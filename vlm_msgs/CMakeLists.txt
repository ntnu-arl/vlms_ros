cmake_minimum_required(VERSION 3.16)
project(vlm_msgs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(semantic_inference_msgs REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  msg/RoomClassificationOutput.msg
  msg/TaskParsingOutput.msg
  srv/TaskParsing.srv
  DEPENDENCIES
  builtin_interfaces
  std_msgs
  semantic_inference_msgs
)

ament_package()
