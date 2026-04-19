cmake_minimum_required(VERSION 3.16)
project(RoadPavementDetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# ── OpenCV ──────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ── ONNX Runtime ─────────────────────────────────────────────────────────────
# Set ONNXRUNTIME_DIR to your ONNX Runtime install prefix, or pass via cmake flag:
#   cmake .. -DONNXRUNTIME_DIR=/usr/local/onnxruntime
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime root")

if(ONNXRUNTIME_DIR)
    include_directories(${ONNXRUNTIME_DIR}/include)
    link_directories(${ONNXRUNTIME_DIR}/lib)
else()
    find_path(ORT_INCLUDE onnxruntime/core/session/onnxruntime_cxx_api.h
              HINTS /usr/local/include /usr/include)
    if(ORT_INCLUDE)
        include_directories(${ORT_INCLUDE})
    else()
        message(FATAL_ERROR "ONNX Runtime headers not found. Set -DONNXRUNTIME_DIR=...")
    endif()
endif()

# ── nlohmann/json (header-only) ──────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(road_detector detector.cpp)

target_link_libraries(road_detector
    ${OpenCV_LIBS}
    onnxruntime
    nlohmann_json::nlohmann_json
)

# Optimize for release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(road_detector PRIVATE -O3 -march=native)
endif()

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV     : ${OpenCV_VERSION}")
