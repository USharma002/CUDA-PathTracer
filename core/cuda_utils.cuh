#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// ============================================================================
// CUDA ERROR CHECKING UTILITIES
// ============================================================================

namespace cuda_utils {

/**
 * @brief Check CUDA error and print message if error occurred
 * @param err CUDA error code
 * @param msg Error message context
 */
inline void checkError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error [" << msg << "]: " << cudaGetErrorString(err) << std::endl;
    }
}

/**
 * @brief Check CUDA error and throw exception if error occurred
 * @param err CUDA error code
 * @param msg Error message context
 */
inline void checkErrorThrow(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::string error_msg = std::string("CUDA Error [") + msg + "]: " + cudaGetErrorString(err);
        throw std::runtime_error(error_msg);
    }
}

/**
 * @brief Safe CUDA memory allocation with error handling
 * @tparam T Type of data to allocate
 * @param ptr Pointer to device memory
 * @param size Size in bytes to allocate
 * @param name Name for error reporting
 */
template<typename T>
inline void safeMalloc(T** ptr, size_t size, const char* name) {
    cudaError_t err = cudaMalloc((void**)ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate " << name << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation failed");
    }
}

/**
 * @brief Safe CUDA memory free
 * @tparam T Type of data to free
 * @param ptr Pointer to device memory
 */
template<typename T>
inline void safeFree(T*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

/**
 * @brief Copy data from host to device
 * @tparam T Type of data
 * @param dst Device destination
 * @param src Host source
 * @param count Number of elements
 */
template<typename T>
inline void copyToDevice(T* dst, const T* src, size_t count) {
    cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
}

/**
 * @brief Copy data from device to host
 * @tparam T Type of data
 * @param dst Host destination
 * @param src Device source
 * @param count Number of elements
 */
template<typename T>
inline void copyToHost(T* dst, const T* src, size_t count) {
    cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost);
}

/**
 * @brief Calculate optimal CUDA grid dimensions
 * @param width Image width
 * @param height Image height
 * @param block_size Block dimensions
 * @return Grid dimensions
 */
inline dim3 calculateGrid(int width, int height, dim3 block_size) {
    return dim3(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );
}

/**
 * @brief Get CUDA device properties
 * @param device_id Device ID (default 0)
 * @return Device properties
 */
inline cudaDeviceProp getDeviceProperties(int device_id = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop;
}

/**
 * @brief Print CUDA device information
 * @param device_id Device ID (default 0)
 */
inline void printDeviceInfo(int device_id = 0) {
    cudaDeviceProp prop = getDeviceProperties(device_id);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
}

} // namespace cuda_utils

// Backward compatibility macros
#define checkCudaError(err, msg) cuda_utils::checkError(err, msg)
#define cudaMallocSafe(ptr, size, name) cuda_utils::safeMalloc(ptr, size, name)

#endif // CUDA_UTILS_CUH
