cmake_minimum_required(VERSION 3.10)
project(TnsrLang)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Add all source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Main executable
add_executable(tnsrlang ${SOURCES})

# Include directory
target_include_directories(tnsrlang PRIVATE src)
