/**
 * @file optix_logger.h
 * @brief Comprehensive logging system for OptiX operations
 * 
 * Provides detailed logging for:
 * - OptiX initialization and context creation
 * - Acceleration structure building
 * - Pipeline creation and shader compilation
 * - Memory allocation and transfers
 * - Ray tracing performance metrics
 */

#ifndef OPTIX_LOGGER_H
#define OPTIX_LOGGER_H

#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <string>
#include <fstream>
#include <mutex>

// ============================================================================
// LOG LEVELS
// ============================================================================

enum class OptixLogLevel {
    TRACE = 0,    // Very detailed tracing (per-ray info, etc.)
    DEBUG = 1,    // Debug info (memory allocations, etc.)
    INFO = 2,     // General info (initialization, build times)
    WARNING = 3,  // Warnings (fallbacks, performance issues)
    ERROR = 4,    // Errors (failures that may be recoverable)
    FATAL = 5,    // Fatal errors (unrecoverable)
    NONE = 6      // Disable logging
};

// ============================================================================
// OPTIX LOGGER CLASS
// ============================================================================

class OptixLogger {
public:
    static OptixLogger& getInstance() {
        static OptixLogger instance;
        return instance;
    }
    
    // Set minimum log level
    void setLogLevel(OptixLogLevel level) { m_log_level = level; }
    OptixLogLevel getLogLevel() const { return m_log_level; }
    
    // Enable/disable file logging
    void setFileLogging(bool enable, const std::string& filename = "optix.log") {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (enable && !m_file.is_open()) {
            m_file.open(filename, std::ios::out | std::ios::trunc);
            if (m_file.is_open()) {
                m_file_logging = true;
                log(OptixLogLevel::INFO, "OptiX Logger", "File logging started: " + filename);
            }
        } else if (!enable && m_file.is_open()) {
            m_file.close();
            m_file_logging = false;
        }
    }
    
    // Main logging function
    void log(OptixLogLevel level, const std::string& category, const std::string& message) {
        if (level < m_log_level) return;
        
        std::lock_guard<std::mutex> lock(m_mutex);
        
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << "[" << std::put_time(std::localtime(&time), "%H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
        ss << "[" << getLevelString(level) << "] ";
        ss << "[" << category << "] ";
        ss << message;
        
        std::string formatted = ss.str();
        
        // Color output for console
        if (level >= OptixLogLevel::ERROR) {
            std::cerr << "\033[1;31m" << formatted << "\033[0m" << std::endl;  // Red
        } else if (level == OptixLogLevel::WARNING) {
            std::cout << "\033[1;33m" << formatted << "\033[0m" << std::endl;  // Yellow
        } else if (level == OptixLogLevel::INFO) {
            std::cout << "\033[1;36m" << formatted << "\033[0m" << std::endl;  // Cyan
        } else {
            std::cout << formatted << std::endl;
        }
        
        // File output
        if (m_file_logging && m_file.is_open()) {
            m_file << formatted << std::endl;
            m_file.flush();
        }
    }
    
    // Convenience methods
    void trace(const std::string& cat, const std::string& msg) { log(OptixLogLevel::TRACE, cat, msg); }
    void debug(const std::string& cat, const std::string& msg) { log(OptixLogLevel::DEBUG, cat, msg); }
    void info(const std::string& cat, const std::string& msg) { log(OptixLogLevel::INFO, cat, msg); }
    void warning(const std::string& cat, const std::string& msg) { log(OptixLogLevel::WARNING, cat, msg); }
    void error(const std::string& cat, const std::string& msg) { log(OptixLogLevel::ERROR, cat, msg); }
    void fatal(const std::string& cat, const std::string& msg) { log(OptixLogLevel::FATAL, cat, msg); }
    
    // Performance logging helpers
    void logMemoryAlloc(const std::string& name, size_t bytes) {
        std::stringstream ss;
        ss << "Allocated " << formatBytes(bytes) << " for " << name;
        debug("Memory", ss.str());
    }
    
    void logMemoryFree(const std::string& name) {
        debug("Memory", "Freed: " + name);
    }
    
    void logBuildTime(const std::string& operation, double ms) {
        std::stringstream ss;
        ss << operation << " completed in " << std::fixed << std::setprecision(2) << ms << " ms";
        info("Performance", ss.str());
    }
    
    void logRayStats(long long rays_traced, double time_ms) {
        std::stringstream ss;
        double mrays_per_sec = (rays_traced / 1e6) / (time_ms / 1000.0);
        ss << "Ray tracing: " << (rays_traced / 1e6) << "M rays in " 
           << std::fixed << std::setprecision(2) << time_ms << " ms ("
           << std::setprecision(1) << mrays_per_sec << " MRays/s)";
        info("Performance", ss.str());
    }
    
    // OptiX callback adapter
    static void optixCallback(unsigned int level, const char* tag, const char* message, void*) {
        auto& logger = getInstance();
        
        OptixLogLevel log_level;
        switch (level) {
            case 1: log_level = OptixLogLevel::FATAL; break;
            case 2: log_level = OptixLogLevel::ERROR; break;
            case 3: log_level = OptixLogLevel::WARNING; break;
            case 4: log_level = OptixLogLevel::INFO; break;
            default: log_level = OptixLogLevel::DEBUG; break;
        }
        
        std::string cat = tag ? std::string(tag) : "OptiX";
        std::string msg = message ? std::string(message) : "";
        
        // Trim trailing newlines
        while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
            msg.pop_back();
        }
        
        logger.log(log_level, cat, msg);
    }
    
private:
    OptixLogger() : m_log_level(OptixLogLevel::INFO), m_file_logging(false) {}
    ~OptixLogger() {
        if (m_file.is_open()) m_file.close();
    }
    
    const char* getLevelString(OptixLogLevel level) {
        switch (level) {
            case OptixLogLevel::TRACE: return "TRACE";
            case OptixLogLevel::DEBUG: return "DEBUG";
            case OptixLogLevel::INFO: return "INFO ";
            case OptixLogLevel::WARNING: return "WARN ";
            case OptixLogLevel::ERROR: return "ERROR";
            case OptixLogLevel::FATAL: return "FATAL";
            default: return "?????";
        }
    }
    
    std::string formatBytes(size_t bytes) {
        std::stringstream ss;
        if (bytes >= 1024 * 1024 * 1024) {
            ss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GB";
        } else if (bytes >= 1024 * 1024) {
            ss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0)) << " MB";
        } else if (bytes >= 1024) {
            ss << std::fixed << std::setprecision(2) << (bytes / 1024.0) << " KB";
        } else {
            ss << bytes << " bytes";
        }
        return ss.str();
    }
    
    OptixLogLevel m_log_level;
    bool m_file_logging;
    std::ofstream m_file;
    std::mutex m_mutex;
};

// ============================================================================
// CONVENIENCE MACROS
// ============================================================================

#define OPTIX_LOG_TRACE(msg) OptixLogger::getInstance().trace("OptiX", msg)
#define OPTIX_LOG_DEBUG(msg) OptixLogger::getInstance().debug("OptiX", msg)
#define OPTIX_LOG_INFO(msg) OptixLogger::getInstance().info("OptiX", msg)
#define OPTIX_LOG_WARNING(msg) OptixLogger::getInstance().warning("OptiX", msg)
#define OPTIX_LOG_ERROR(msg) OptixLogger::getInstance().error("OptiX", msg)
#define OPTIX_LOG_FATAL(msg) OptixLogger::getInstance().fatal("OptiX", msg)

// Scoped timer for automatic performance logging
class OptixScopedTimer {
public:
    OptixScopedTimer(const std::string& name) : m_name(name) {
        m_start = std::chrono::high_resolution_clock::now();
    }
    
    ~OptixScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - m_start).count();
        OptixLogger::getInstance().logBuildTime(m_name, ms);
    }
    
private:
    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

#define OPTIX_TIMED_SCOPE(name) OptixScopedTimer _timer_##__LINE__(name)

#endif // OPTIX_LOGGER_H
