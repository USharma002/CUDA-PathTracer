cmake_minimum_required(VERSION 3.18) # good CUDA support
project(CUDAPathTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# ============================================================================
# Options
# ============================================================================
option(ENABLE_OPTIX "Enable OptiX support for hardware-accelerated ray tracing" OFF)
option(BUILD_TESTS "Build test executables" OFF)

# ============================================================================
# Dependencies via vcpkg
# ============================================================================
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

# ============================================================================
# OptiX (optional)
# ============================================================================
if(ENABLE_OPTIX)
    # Try to find OptiX
    find_path(OptiX_INCLUDE_DIR
        NAMES optix.h
        PATHS
            $ENV{OptiX_INSTALL_DIR}/include
            /usr/local/optix/include
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK*/include"
        DOC "OptiX include directory"
    )
    
    if(OptiX_INCLUDE_DIR)
        message(STATUS "Found OptiX: ${OptiX_INCLUDE_DIR}")
        add_definitions(-DENABLE_OPTIX=1)
    else()
        message(WARNING "OptiX not found. Disabling OptiX support.")
        set(ENABLE_OPTIX OFF)
    endif()
endif()

# ============================================================================
# ImGui sources
# ============================================================================
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/ext/imgui)
set(EXT_DIR ${CMAKE_SOURCE_DIR}/ext/)
set(IMGUI_SRC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/imgui_impl_opengl3.cpp
    ${EXT_DIR}/ImGuiFileDialog/ImGuiFileDialog.cpp
)

# Create imgui static lib
add_library(imgui STATIC ${IMGUI_SRC})
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
)
target_link_libraries(imgui
    PUBLIC
        glfw
        GLEW::GLEW
        OpenGL::GL
)

# ============================================================================
# Core Module Headers (header-only)
# ============================================================================
set(CORE_HEADERS
    ${CMAKE_SOURCE_DIR}/core/cuda_utils.cuh
    ${CMAKE_SOURCE_DIR}/core/app_config.cuh
    ${CMAKE_SOURCE_DIR}/core/render_state.cuh
    ${CMAKE_SOURCE_DIR}/core/scene_state.cuh
    ${CMAKE_SOURCE_DIR}/core/radiosity_state.cuh
    ${CMAKE_SOURCE_DIR}/core/ui_state.cuh
)

set(RENDERING_HEADERS
    ${CMAKE_SOURCE_DIR}/rendering/path_tracer.cuh
    ${CMAKE_SOURCE_DIR}/rendering/radiosity_renderer.cuh
)

set(UI_HEADERS
    ${CMAKE_SOURCE_DIR}/ui/controls_window.cuh
    ${CMAKE_SOURCE_DIR}/ui/grid_window.cuh
)

set(ACCEL_HEADERS
    ${CMAKE_SOURCE_DIR}/accel/optix_types.cuh
    ${CMAKE_SOURCE_DIR}/accel/optix_wrapper.cuh
)

# ============================================================================
# Main executable
# ============================================================================
set(MAIN_SOURCES
    main.cu
)

# Add OptiX programs if enabled
if(ENABLE_OPTIX)
    list(APPEND MAIN_SOURCES accel/optix_programs.cu)
endif()

add_executable(CUDAPathTracer
    ${MAIN_SOURCES}
    ${CORE_HEADERS}
    ${RENDERING_HEADERS}
    ${UI_HEADERS}
    ${ACCEL_HEADERS}
)

# CUDA settings
set_target_properties(CUDAPathTracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
    CUDA_ARCHITECTURES "60;70;75;80;86"  # Support multiple GPU architectures
)

# Include directories
target_include_directories(CUDAPathTracer PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/rendering
    ${CMAKE_SOURCE_DIR}/ui
    ${CMAKE_SOURCE_DIR}/accel
    ${CMAKE_SOURCE_DIR}/ext/imgui
    ${CMAKE_SOURCE_DIR}/ext/imgui/backends
    ${CMAKE_SOURCE_DIR}/ext/ImGuiFileDialog
)

# OptiX include directory
if(ENABLE_OPTIX)
    target_include_directories(CUDAPathTracer PRIVATE ${OptiX_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(CUDAPathTracer
    PRIVATE
        imgui
        glfw
        GLEW::GLEW
        OpenGL::GL
)

# Windows-specific
if(WIN32)
    target_link_libraries(CUDAPathTracer PRIVATE opengl32)
endif()

# ============================================================================
# Compile definitions
# ============================================================================
target_compile_definitions(CUDAPathTracer PRIVATE
    $<$<CONFIG:Debug>:DEBUG_BUILD>
    $<$<CONFIG:Release>:NDEBUG>
)

if(ENABLE_OPTIX)
    target_compile_definitions(CUDAPathTracer PRIVATE ENABLE_OPTIX=1)
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS CUDAPathTracer DESTINATION bin)
install(DIRECTORY scenes/ DESTINATION scenes)
install(DIRECTORY assets/ DESTINATION assets)

# ============================================================================
# Copy assets and scenes to build directory for development
# ============================================================================
add_custom_command(TARGET CUDAPathTracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/scenes
        $<TARGET_FILE_DIR:CUDAPathTracer>/scenes
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:CUDAPathTracer>/assets
    COMMENT "Copying scenes and assets to build directory"
)

# ============================================================================
# Print configuration summary
# ============================================================================
message(STATUS "")
message(STATUS "=== CUDA Path Tracer Configuration ===")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  OptiX support: ${ENABLE_OPTIX}")
message(STATUS "  CUDA architectures: 60;70;75;80;86")
message(STATUS "=======================================")
message(STATUS "")
