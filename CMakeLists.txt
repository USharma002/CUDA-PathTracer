cmake_minimum_required(VERSION 3.18) # good CUDA support
project(TestImGui LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

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
    main.cu   # your CUDA entry point
)

# CUDA settings
set_target_properties(TestImGui PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)

target_include_directories(TestImGui PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ext/imgui
    ${CMAKE_SOURCE_DIR}/ext/imgui/backends
    ${CMAKE_SOURCE_DIR}/ext/ImGuiFileDialog
)

target_link_libraries(TestImGui
    PRIVATE
        imgui
        glfw
        GLEW::GLEW     # or glad::glad
        OpenGL::GL
)

# On Windows we must also link against opengl32.lib
if(WIN32)
    target_link_libraries(TestImGui PRIVATE opengl32)
endif()
