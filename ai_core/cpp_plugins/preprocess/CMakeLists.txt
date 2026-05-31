cmake_minimum_required(VERSION 3.10)
project(yolox_preprocess)

# Enable CUDA and C++ dialects
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Standard C++ and CUDA compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -O3 -fPIC -shared")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC -shared")

# Add the compiled object library
add_library(${PROJECT_NAME} SHARED 
    yolox_preprocess.cu
)

# DeepStream / CUDA Includes
target_include_directories(${PROJECT_NAME} PRIVATE 
    /usr/local/cuda/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against CUDA dynamic library
target_link_libraries(${PROJECT_NAME}
    cudart
)
