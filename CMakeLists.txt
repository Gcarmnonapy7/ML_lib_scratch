cmake_minimum_required(VERSION 3.14)

project(My_Machine_Learning_Scratch VERSION 1.0 
    DESCRIPTION "A collection of machine learning algorithms implemented from scratch"
    LANGUAGES CXX)

# Set C++ standard to C++ 17 or higher
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warning to the compliers 
if (MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include directories for headers
include_directories(${PROJECT_SOURCE_DIR}/include)

#Source files 

# set(SOURCES 
#     src/knn.cpp
#     #src/bindings.cpp only when i finish the first implementation of the lib
#     )

#Static lib at first
add_library(my_ml_lib STATIC${SOURCES})

#Adding executable for testing 
add_executable(my_ml_lib_test src/main.cpp)

target_link_libraries(my_ml_lib_test PRIVATE my_ml_lib)


# For Pybind11 (later)
# find_package(pybind11 REQUIRED)
# target_link_libraries(my_ml_lib PRIVATE pybind11::module)