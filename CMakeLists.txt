cmake_minimum_required(VERSION 3.18) # good CUDA support
project(TestImGui LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# ---- OptiX Support (Optional) ----
option(USE_OPTIX "Enable OptiX ray tracing acceleration" ON)
option(USE_PBRT_LOADER "Enable PBRT scene file loading" ON)

if(USE_OPTIX)
    # Find OptiX SDK - search for multiple versions (prefer newer)
    find_path(OptiX_INCLUDE_DIR
        NAMES optix.h
        PATHS
            "$ENV{OptiX_INSTALL_DIR}/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.6.0/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0/include"
            "/opt/optix/include"
            "/usr/local/optix/include"
    )
    
    if(OptiX_INCLUDE_DIR)
        message(STATUS "===========================================")
        message(STATUS "Found OptiX: ${OptiX_INCLUDE_DIR}")
        message(STATUS "OptiX hardware ray tracing ENABLED")
        message(STATUS "===========================================")
        add_compile_definitions(USE_OPTIX)
    else()
        message(WARNING "OptiX SDK not found - building without OptiX support")
        message(WARNING "Install OptiX SDK from https://developer.nvidia.com/optix")
        set(USE_OPTIX OFF)
    endif()
endif()

# ---- PBRT Parser Library (Optional) ----
if(USE_PBRT_LOADER)
    set(PBRT_PARSER_DIR ${CMAKE_SOURCE_DIR}/ext/pbrtparser)
    if(EXISTS ${PBRT_PARSER_DIR}/CMakeLists.txt)
        message(STATUS "===========================================")
        message(STATUS "PBRT Loader ENABLED")
        message(STATUS "===========================================")
        add_compile_definitions(USE_PBRT_LOADER)
        
        # Disable apps and unittests - we just want the library
        set(pbrtParser_BUILD_APPS OFF CACHE BOOL "" FORCE)
        set(pbrtParser_ENABLE_UNITTESTS OFF CACHE BOOL "" FORCE)
        
        # Add pbrtParser as subdirectory - it creates 'pbrtParser' target
        add_subdirectory(${PBRT_PARSER_DIR} ${CMAKE_BINARY_DIR}/pbrtparser)
    else()
        message(WARNING "PBRT Parser not found at ${PBRT_PARSER_DIR}")
        message(WARNING "Building without PBRT support")
        set(USE_PBRT_LOADER OFF)
    endif()
endif()

# ---- Dependencies via vcpkg ----
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)   # or glad if you prefer
find_package(OpenGL REQUIRED)

# ---- ImGui sources ----
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

# Link imgui against glfw + glew/glad + OpenGL
target_link_libraries(imgui
    PUBLIC
        glfw
        GLEW::GLEW     # or glad::glad if using glad
        OpenGL::GL
)

# ---- Your main project ----
add_executable(TestImGui
    src/main.cu   # your CUDA entry point
)

# CUDA settings
set_target_properties(TestImGui PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)

target_include_directories(TestImGui PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/core
    ${CMAKE_SOURCE_DIR}/include/rendering
    ${CMAKE_SOURCE_DIR}/include/ui
    ${CMAKE_SOURCE_DIR}/include/utils
    ${CMAKE_SOURCE_DIR}/ext/imgui
    ${CMAKE_SOURCE_DIR}/ext/imgui/backends
    ${CMAKE_SOURCE_DIR}/ext/ImGuiFileDialog
)

# Add OptiX include if available
if(USE_OPTIX AND OptiX_INCLUDE_DIR)
    target_include_directories(TestImGui PRIVATE ${OptiX_INCLUDE_DIR})
endif()

target_link_libraries(TestImGui
    PRIVATE
        imgui
        glfw
        GLEW::GLEW     # or glad::glad
        OpenGL::GL
)

# Link pbrtParser if enabled (target name from pbrtParser's CMakeLists.txt)
if(USE_PBRT_LOADER)
    target_link_libraries(TestImGui PRIVATE pbrtParser)
endif()

# On Windows we must also link against opengl32.lib
if(WIN32)
    target_link_libraries(TestImGui PRIVATE opengl32)
endif()
