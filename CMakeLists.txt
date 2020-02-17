cmake_minimum_required(VERSION 3.15)
project(machine_learning_lib)

set(CMAKE_CXX_STANDARD 14)

add_library(machine_learning_lib SHARED library.cpp library.h)