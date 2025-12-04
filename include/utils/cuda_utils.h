/**
 * @file cuda_utils.h
 * @brief CUDA utility functions and macros
 * 
 * Contains helper functions for CUDA memory management and error checking.
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// ============================================================================
// ERROR CHECKING
// ============================================================================

/**
 * @brief Check CUDA error and print message if error occurred
 * @param err CUDA error code
 * @param msg Context message for the error
 */
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error [" << msg << "]: " << cudaGetErrorString(err) << std::endl;
    }
}

/**
 * @brief Check CUDA error and throw exception on failure
 * @param err CUDA error code
 * @param msg Context message for the error
 */
inline void checkCudaErrorFatal(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Fatal Error [" << msg << "]: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA operation failed");
    }
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

/**
 * @brief Safe CUDA memory allocation with error checking
 * @tparam T Type of data to allocate
 * @param ptr Pointer to device memory (output)
 * @param size Size in bytes to allocate
 * @param name Name for error reporting
 */
template<typename T>
inline void cudaMallocSafe(T** ptr, size_t size, const char* name) {
    cudaError_t err = cudaMalloc((void**)ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate " << name << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
}

/**
 * @brief Safe CUDA memory free
 * @tparam T Type of data to free
 * @param ptr Pointer to device memory (will be set to nullptr)
 */
template<typename T>
inline void cudaFreeSafe(T*& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

// ============================================================================
// GRID/BLOCK HELPERS
// ============================================================================

/**
 * @brief Calculate grid size for a given data size and block size
 * @param data_size Total number of elements
 * @param block_size Number of threads per block
 * @return Number of blocks needed
 */
inline int calcGridSize(int data_size, int block_size) {
    return (data_size + block_size - 1) / block_size;
}

/**
 * @brief Calculate 2D grid dimensions
 * @param width Width of data
 * @param height Height of data
 * @param block Block dimensions
 * @return Grid dimensions
 */
inline dim3 calcGrid2D(int width, int height, dim3 block) {
    return dim3(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
}

#endif // CUDA_UTILS_H
