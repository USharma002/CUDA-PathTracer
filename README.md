# CUDA Path Tracer

A GPU-accelerated **path tracer** implemented in CUDA with optional **OptiX** support for hardware-accelerated ray tracing on RTX GPUs.

It renders photorealistic images using **Monte Carlo integration**, **Russian roulette termination**, and support for **materials** loaded from `.obj` and `.mtl` files.

This implementation is capable of rendering complex scenes interactively or offline, leveraging CUDA's massive parallelism for efficient global illumination computation.

API description: [Rendering Engine Docs](./media/API.md)

## Example Render (Cornell Box)

![Cornell Box Render](assets/cbox.png)

## Features

- **CUDA-accelerated** ray tracing and shading
- **OptiX support** (optional) for hardware-accelerated BVH traversal on RTX GPUs
- Support for diffuse, reflective, and emissive materials
- `.obj` geometry and `.mtl` material loading
- Configurable samples-per-pixel (SPP) and max bounce depth
- **Radiosity solver** with form factor computation
- Multiple sampling strategies (BSDF, Grid, MIS)
- **Modular architecture** for easy extension

## Project Structure

```
CUDA-PathTracer/
├── main.cu                    # Main application entry point
├── CMakeLists.txt            # CMake build configuration
├── core/                     # Core modules
│   ├── cuda_utils.cuh        # CUDA utility functions
│   ├── app_config.cuh        # Application configuration
│   ├── render_state.cuh      # Render state management
│   ├── scene_state.cuh       # Scene state management
│   ├── radiosity_state.cuh   # Radiosity solver state
│   └── ui_state.cuh          # UI state management
├── rendering/                # Rendering modules
│   ├── path_tracer.cuh       # Path tracing integrator
│   └── radiosity_renderer.cuh # Radiosity renderer
├── accel/                    # Acceleration structures
│   ├── optix_types.cuh       # OptiX type definitions
│   ├── optix_wrapper.cuh     # OptiX wrapper class
│   └── optix_programs.cu     # OptiX ray tracing programs
├── ui/                       # User interface modules
│   ├── controls_window.cuh   # Main controls window
│   └── grid_window.cuh       # Grid visualization window
├── scenes/                   # Scene files (.obj, .mtl)
├── assets/                   # Assets (images, etc.)
└── ext/                      # External dependencies (ImGui, etc.)
```

## Building & Running

### Prerequisites

- **CUDA Toolkit** 11.0 or later
- **CMake** 3.18 or later
- **vcpkg** for dependency management
- **GLFW3**, **GLEW**, **OpenGL** (installed via vcpkg)
- **OptiX SDK** 7.5+ (optional, for hardware-accelerated ray tracing)

### Windows Build

```batch
# Standard build
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

# With OptiX support
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DENABLE_OPTIX=ON ^
  -DOptiX_INSTALL_DIR="C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0"
cmake --build build --config Release
```

Or simply run `build.bat` for a standard build.

### Linux Build

```bash
# Standard build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# With OptiX support
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPTIX=ON \
  -DOptiX_INSTALL_DIR=/usr/local/optix
make -j$(nproc)
```

## Usage

### Camera Controls
- **Left click + drag**: Rotate camera (orbit)
- **Scroll wheel**: Zoom in/out

### UI Controls
- **Resolution**: Adjust render resolution
- **SPP**: Samples per pixel (higher = better quality, slower)
- **Integrator**: Choose between Path Tracing, Radiosity, or Delta visualization
- **Sampling Mode**: BSDF, Grid-based, or MIS (Multiple Importance Sampling)
- **Radiosity**: Configure and run radiosity solver

## OptiX Integration

When built with `ENABLE_OPTIX=ON`, the path tracer can use NVIDIA OptiX for hardware-accelerated BVH construction and ray traversal. This provides significant performance improvements on RTX GPUs:

- **Hardware BVH**: Utilizes RT cores for acceleration structure traversal
- **Faster frame rates**: 2-10x speedup depending on scene complexity
- **Same quality**: Produces identical results to the standard CUDA path

To use OptiX:
1. Install the [NVIDIA OptiX SDK](https://developer.nvidia.com/optix)
2. Build with `-DENABLE_OPTIX=ON`
3. The application will automatically use OptiX when available

## License

MIT License
