cmake_minimum_required(VERSION 3.10)
project(ocsort_api)

# C++ Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIC")

# Find Eigen3 (Header-only library)
find_package(Eigen3 REQUIRED NO_MODULE)

# Collect all source files
set(SOURCES
    src/OCSortAPI.cpp
    src/OCSort.cpp
    src/KalmanBoxTracker.cpp
    src/KalmanFilter.cpp
    src/Association.cpp
    src/Utilities.cpp
    src/lapjv.cpp
)

# Create the shared library (.so) for Python ctypes loading
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Link Eigen3
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# Ensure the output library is named libocsort_api.so
set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "ocsort_api"
    PREFIX "lib"
)
