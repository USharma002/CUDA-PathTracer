@echo off
echo ========================================
echo Building CUDA PathTracer...
echo ========================================

:: Configure CMake if build directory doesn't exist or needs refresh
if not exist "build\CMakeCache.txt" (
    echo Configuring CMake...
    cmake -B build -S . -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake
    if errorlevel 1 (
        echo CMake configuration failed!
        exit /b 1
    )
)

:: Build
echo Building...
cmake --build build --config Release
if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

:: Copy executable and required DLLs to root for easy running
echo.
echo Copying files to root directory...
copy /Y "build\Release\TestImGui.exe" "." >nul
copy /Y "build\Release\glew32.dll" "." >nul
copy /Y "build\Release\glfw3.dll" "." >nul

echo.
echo ========================================
echo Build successful!
echo Run: TestImGui.exe
echo ========================================