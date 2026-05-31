cmake_minimum_required(VERSION 3.10)
project(libocsort)

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od") 
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0") 
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_RELEASE} -O3")
endif()

set(CMAKE_CXX_STANDARD 17)

# Find external libraries
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
file(GLOB SRC_LIST src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# note：test with yolo inference
# add_executable(test mutilthread.cpp ../detector/inference.h ../detector/inference.cpp)
# if(UNIX)
#     target_link_libraries(test PUBLIC Eigen3::Eigen ${PROJECT_NAME} ${OpenCV_LIBS} avcodec avformat avutil swscale)
# else()
#     target_link_libraries(test PUBLIC Eigen3::Eigen ${PROJECT_NAME} ${OpenCV_LIBS})
# endif()

