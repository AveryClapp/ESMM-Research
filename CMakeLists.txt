cmake_minimum_required(VERSION 3.18)
project(ESMM_Research LANGUAGES CXX CUDA)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)  # Change to your GPU architecture

# Find CUDA libraries
find_package(CUDAToolkit REQUIRED)

# Source files
set(DRIVER_SOURCE driver.cu)

# All kernel and header dependencies
file(GLOB_RECURSE KERNEL_SOURCES
    "src/kernels/*.cu"
    "src/preprocessors/*.cu"
    "old_kernels/*.cu"
)

file(GLOB_RECURSE HEADERS
    "include/*.cuh"
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Development build (default): fast compilation
add_executable(driver ${DRIVER_SOURCE})
target_link_libraries(driver CUDA::cublas CUDA::cusparse)
target_compile_options(driver PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O1>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Release build: optimized
add_executable(driver_release ${DRIVER_SOURCE})
target_link_libraries(driver_release CUDA::cublas CUDA::cusparse)
target_compile_options(driver_release PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Quick build: no optimization
add_executable(driver_quick ${DRIVER_SOURCE})
target_link_libraries(driver_quick CUDA::cublas CUDA::cusparse)
target_compile_options(driver_quick PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O0>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Profile build: optimized with lineinfo for NCU profiling
add_executable(driver_profile ${DRIVER_SOURCE})
target_link_libraries(driver_profile CUDA::cublas CUDA::cusparse)
target_compile_options(driver_profile PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Add dependencies so CMake knows when to recompile
# Note: Since headers include .cu files, they're all dependencies
set_source_files_properties(${DRIVER_SOURCE} PROPERTIES
    OBJECT_DEPENDS "${KERNEL_SOURCES};${HEADERS}"
)

# NCU profiling custom targets
set(NCU_FLAGS "--set full --target-processes all")
set(NCU_OUTPUT "profile.ncu-rep")

# NCU profile: Build + profile with full metrics
add_custom_target(ncu-profile
    COMMAND bash -c "\
        NCU_CMD='sudo ncu'; \
        if [ -n \"$$NO_SUDO\" ]; then NCU_CMD='ncu'; fi; \
        echo 'Profiling kernel(s)' $$KERNEL'...'; \
        echo 'Output: ${NCU_OUTPUT}'; \
        if [ -z \"$$KERNEL\" ]; then \
            echo 'Error: Please specify KERNEL=<selection>'; \
            echo 'Examples:'; \
            echo '  KERNEL=17 cmake --build . --target ncu-profile'; \
            echo '  KERNEL=17,18,19 cmake --build . --target ncu-profile'; \
            echo '  KERNEL=17-20 cmake --build . --target ncu-profile'; \
            echo '  NO_SUDO=1 KERNEL=17 cmake --build . --target ncu-profile'; \
            exit 1; \
        fi; \
        $$NCU_CMD ${NCU_FLAGS} -o ${NCU_OUTPUT} ./driver_profile $$KERNEL 100"
    DEPENDS driver_profile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running NCU profiler with full metrics"
)

# NCU run: Profile without rebuilding
add_custom_target(ncu-run
    COMMAND bash -c "\
        NCU_CMD='sudo ncu'; \
        if [ -n \"$$NO_SUDO\" ]; then NCU_CMD='ncu'; fi; \
        echo 'Profiling kernel(s)' $$KERNEL'...'; \
        echo 'Output: ${NCU_OUTPUT}'; \
        if [ ! -f driver_profile ]; then \
            echo 'Error: driver_profile not found. Build it first.'; \
            exit 1; \
        fi; \
        if [ -z \"$$KERNEL\" ]; then \
            echo 'Error: Please specify KERNEL=<selection>'; \
            exit 1; \
        fi; \
        $$NCU_CMD ${NCU_FLAGS} -o ${NCU_OUTPUT} ./driver_profile $$KERNEL 100"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running NCU profiler (no rebuild)"
)

# NCU quick: Build + quick profile (default metrics)
add_custom_target(ncu-quick
    COMMAND bash -c "\
        NCU_CMD='sudo ncu'; \
        if [ -n \"$$NO_SUDO\" ]; then NCU_CMD='ncu'; fi; \
        echo 'Quick profiling kernel(s)' $$KERNEL'...'; \
        echo 'Output: ${NCU_OUTPUT}'; \
        if [ -z \"$$KERNEL\" ]; then \
            echo 'Error: Please specify KERNEL=<selection>'; \
            exit 1; \
        fi; \
        $$NCU_CMD --target-processes all -o ${NCU_OUTPUT} ./driver_profile $$KERNEL 100"
    DEPENDS driver_profile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running NCU profiler (quick mode)"
)

# Print configuration
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  driver         - Development build (-O1, fast compile)")
message(STATUS "  driver_release - Production build (-O3, optimized)")
message(STATUS "  driver_quick   - Quick test build (-O0, fastest compile)")
message(STATUS "  driver_profile - NCU profiling build (-O3, lineinfo)")
message(STATUS "")
message(STATUS "NCU Profiling targets:")
message(STATUS "  ncu-profile    - Build + profile with --set full")
message(STATUS "  ncu-run        - Profile without rebuild")
message(STATUS "  ncu-quick      - Build + quick profile")
message(STATUS "")
message(STATUS "Usage examples (sudo by default):")
message(STATUS "  KERNEL=17 cmake --build . --target ncu-profile")
message(STATUS "  KERNEL=17,18,19 cmake --build . --target ncu-profile")
message(STATUS "  KERNEL=17-20 cmake --build . --target ncu-run")
message(STATUS "  NO_SUDO=1 KERNEL=17 cmake --build . --target ncu-profile")
