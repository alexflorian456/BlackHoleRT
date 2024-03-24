# ORIGINAL:
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(BlackHoleRT VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(CTest)
enable_testing()

add_executable(${CMAKE_PROJECT_NAME} src/main.cu)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 86)
set_target_properties( ${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --verbose -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr>
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})

set(glfw3_DIR "C:/Program Files (x86)/GLFW/lib/cmake/glfw3")
set(glfw3_INC_DIR "C:/Program Files (x86)/GLFW/include")
find_package(glfw3 3.4 REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} glfw opengl32.lib)
include_directories(${glfw3_INC_DIR})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# for reference, check: https://github.com/Viatreen/AI_Art_Generator
# 
# EXAMPLE
# cmake_minimum_required(VERSION 3.17)

# if (MSVC)
#   set(CudaToolkitDir "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0")
#   set(CMAKE_CUDA_COMPILER "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin\\nvcc.exe")
#   set(CMAKE_CXX_COMPILER "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.34.31933\\bin\\Hostx64\\x64\\cl.exe")
#   set(CUDA_TOOLKIT_ROOT_DIR "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0")
# endif()

# project(opencv-test LANGUAGES CXX CUDA)
# message("Project name: " ${CMAKE_PROJECT_NAME})

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)

# if (MSVC)
#     set(_CRT_SECURE_NO_WARNINGS 1)
#     set(CURL_STATICLIB 1)
# endif()

# include_directories(
#     /
# )

# set(sources
#     main.cu
# )

# add_executable(${CMAKE_PROJECT_NAME} ${sources})

# target_compile_options(${CMAKE_PROJECT_NAME}
#     PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr> # -g -G --ptxas-options=–allow-expensive-optimizations=true>
# )

# set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 86)

# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# endif()