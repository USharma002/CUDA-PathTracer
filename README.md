# CUDA Path Tracer

A GPU-accelerated **path tracer** implemented in CUDA.  
It renders photorealistic images using **Monte Carlo integration**, **Russian roulette termination**, and support for **materials** loaded from `.obj` and `.mtl` files.

This implementation is capable of rendering complex scenes interactively or offline, leveraging CUDA’s massive parallelism for efficient global illumination computation.

## Example Render (Cornell Box)

![Cornell Box Render](assets/cbox.png)

## Features
- CUDA-accelerated ray tracing and shading
- Support for diffuse, reflective, and emissive materials
- `.obj` geometry and `.mtl` material loading
- Configurable samples-per-pixel (SPP) and max bounce depth

## Building & Running
```bash
nvcc main.cu -o rt
./rt
