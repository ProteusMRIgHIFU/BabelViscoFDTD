
cmake_minimum_required(VERSION 3.7)
project(pi_ocl)
find_package(OpenCL REQUIRED)
add_executable(pi_ocl pi_ocl.c)
target_link_libraries(pi_ocl OpenCL::OpenCL)