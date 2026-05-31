cmake_minimum_required(VERSION 3.10)
project(nvdsinfer_custombboxparser_yolox)

# C++ Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -O3 -fPIC -shared")

# Add the DeepStream specific NMS threshold variable if you want to override the default 0.7
# add_definitions(-DYOLOX_NMS_THRESH=0.7)

# Create the shared library (.so)
add_library(${PROJECT_NAME} SHARED 
    nvdsinfer_custombboxparser_yolox.cpp
)

# DeepStream SDK Includes (Adjust path based on your Jetson DS version, generally 6.2 or 6.3)
target_include_directories(${PROJECT_NAME} PRIVATE 
    /opt/nvidia/deepstream/deepstream/sources/includes
    /usr/local/cuda/include
)

# Link against TensorRT and NVInfer if necessary (parser might not strictly need it, but safe to include)
target_link_libraries(${PROJECT_NAME}
    nvinfer
)
