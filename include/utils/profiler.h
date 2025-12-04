/**
 * @file profiler.h
 * @brief GPU/CPU Profiler with CUDA events and ImGui visualization
 * 
 * Features:
 * - GPU timing using CUDA events
 * - CPU timing using std::chrono
 * - Rolling history for graphs
 * - ImGui visualization with histograms
 */

#ifndef PROFILER_H
#define PROFILER_H

#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>

// ============================================================================
// PROFILER CONFIGURATION
// ============================================================================

#define PROFILER_HISTORY_SIZE 120   // Number of frames to keep in history
#define PROFILER_MAX_STAGES 16      // Maximum number of profiling stages

// ============================================================================
// GPU TIMER - Uses CUDA events for accurate GPU timing
// ============================================================================

class GPUTimer {
public:
    GPUTimer() : m_started(false), m_elapsed_ms(0.0f) {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }
    
    ~GPUTimer() {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }
    
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(m_start, stream);
        m_started = true;
    }
    
    void stop(cudaStream_t stream = 0) {
        if (m_started) {
            cudaEventRecord(m_stop, stream);
            m_started = false;
        }
    }
    
    // Call after cudaDeviceSynchronize() or stream sync
    float getElapsedMs() {
        cudaEventSynchronize(m_stop);
        cudaEventElapsedTime(&m_elapsed_ms, m_start, m_stop);
        return m_elapsed_ms;
    }
    
private:
    cudaEvent_t m_start, m_stop;
    bool m_started;
    float m_elapsed_ms;
};

// ============================================================================
// CPU TIMER - Uses high_resolution_clock
// ============================================================================

class CPUTimer {
public:
    CPUTimer() : m_elapsed_ms(0.0f) {}
    
    void start() {
        m_start = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        m_stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(m_stop - m_start);
        m_elapsed_ms = duration.count() / 1000.0f;
    }
    
    float getElapsedMs() const { return m_elapsed_ms; }
    
private:
    std::chrono::high_resolution_clock::time_point m_start, m_stop;
    float m_elapsed_ms;
};

// ============================================================================
// PROFILER STAGE - Tracks timing for a single stage
// ============================================================================

struct ProfilerStage {
    std::string name;
    float current_ms;
    float history[PROFILER_HISTORY_SIZE];
    int history_idx;
    bool is_gpu;  // true = GPU timer, false = CPU timer
    
    GPUTimer gpu_timer;
    CPUTimer cpu_timer;
    
    ProfilerStage(const std::string& n = "", bool gpu = true) 
        : name(n), current_ms(0.0f), history_idx(0), is_gpu(gpu) {
        std::fill(std::begin(history), std::end(history), 0.0f);
    }
    
    void start(cudaStream_t stream = 0) {
        if (is_gpu) gpu_timer.start(stream);
        else cpu_timer.start();
    }
    
    void stop(cudaStream_t stream = 0) {
        if (is_gpu) gpu_timer.stop(stream);
        else cpu_timer.stop();
    }
    
    void recordFrame() {
        current_ms = is_gpu ? gpu_timer.getElapsedMs() : cpu_timer.getElapsedMs();
        history[history_idx] = current_ms;
        history_idx = (history_idx + 1) % PROFILER_HISTORY_SIZE;
    }
    
    float getAverage() const {
        float sum = 0.0f;
        int count = 0;
        for (int i = 0; i < PROFILER_HISTORY_SIZE; i++) {
            if (history[i] > 0.0f) {
                sum += history[i];
                count++;
            }
        }
        return count > 0 ? sum / count : 0.0f;
    }
    
    float getMax() const {
        float max_val = 0.0f;
        for (int i = 0; i < PROFILER_HISTORY_SIZE; i++) {
            max_val = std::max(max_val, history[i]);
        }
        return max_val;
    }
    
    float getMin() const {
        float min_val = FLT_MAX;
        for (int i = 0; i < PROFILER_HISTORY_SIZE; i++) {
            if (history[i] > 0.0f) {
                min_val = std::min(min_val, history[i]);
            }
        }
        return min_val == FLT_MAX ? 0.0f : min_val;
    }
};

// ============================================================================
// PROFILER - Main profiling system
// ============================================================================

class Profiler {
public:
    static Profiler& getInstance() {
        static Profiler instance;
        return instance;
    }
    
    // Register a new profiling stage
    int addStage(const std::string& name, bool is_gpu = true) {
        if (m_stages.size() >= PROFILER_MAX_STAGES) {
            return -1;
        }
        m_stages.emplace_back(name, is_gpu);
        m_stage_map[name] = m_stages.size() - 1;
        return m_stages.size() - 1;
    }
    
    // Get stage by name
    int getStageId(const std::string& name) {
        auto it = m_stage_map.find(name);
        return it != m_stage_map.end() ? it->second : -1;
    }
    
    // Start timing a stage
    void startStage(int stage_id, cudaStream_t stream = 0) {
        if (stage_id >= 0 && stage_id < (int)m_stages.size() && m_enabled) {
            m_stages[stage_id].start(stream);
        }
    }
    
    void startStage(const std::string& name, cudaStream_t stream = 0) {
        startStage(getStageId(name), stream);
    }
    
    // Stop timing a stage
    void stopStage(int stage_id, cudaStream_t stream = 0) {
        if (stage_id >= 0 && stage_id < (int)m_stages.size() && m_enabled) {
            m_stages[stage_id].stop(stream);
        }
    }
    
    void stopStage(const std::string& name, cudaStream_t stream = 0) {
        stopStage(getStageId(name), stream);
    }
    
    // Call at end of frame to record all timings
    void endFrame() {
        if (!m_enabled) return;
        
        // Record all GPU stages (requires sync)
        cudaDeviceSynchronize();
        
        float total_ms = 0.0f;
        for (auto& stage : m_stages) {
            stage.recordFrame();
            total_ms += stage.current_ms;
        }
        
        // Record total frame time
        m_total_frame_ms = total_ms;
        m_frame_history[m_frame_history_idx] = total_ms;
        m_frame_history_idx = (m_frame_history_idx + 1) % PROFILER_HISTORY_SIZE;
        m_frame_count++;
    }
    
    // Enable/disable profiling
    void setEnabled(bool enabled) { m_enabled = enabled; }
    bool isEnabled() const { return m_enabled; }
    
    // Getters
    const std::vector<ProfilerStage>& getStages() const { return m_stages; }
    float getTotalFrameMs() const { return m_total_frame_ms; }
    float getFPS() const { return m_total_frame_ms > 0.0f ? 1000.0f / m_total_frame_ms : 0.0f; }
    const float* getFrameHistory() const { return m_frame_history; }
    int getFrameHistoryIdx() const { return m_frame_history_idx; }
    int getFrameCount() const { return m_frame_count; }
    
    float getAverageFrameTime() const {
        float sum = 0.0f;
        int count = 0;
        for (int i = 0; i < PROFILER_HISTORY_SIZE; i++) {
            if (m_frame_history[i] > 0.0f) {
                sum += m_frame_history[i];
                count++;
            }
        }
        return count > 0 ? sum / count : 0.0f;
    }
    
    // Reset all timings
    void reset() {
        for (auto& stage : m_stages) {
            std::fill(std::begin(stage.history), std::end(stage.history), 0.0f);
            stage.history_idx = 0;
            stage.current_ms = 0.0f;
        }
        std::fill(std::begin(m_frame_history), std::end(m_frame_history), 0.0f);
        m_frame_history_idx = 0;
        m_frame_count = 0;
    }
    
private:
    Profiler() : m_enabled(true), m_total_frame_ms(0.0f), 
                 m_frame_history_idx(0), m_frame_count(0) {
        std::fill(std::begin(m_frame_history), std::end(m_frame_history), 0.0f);
    }
    
    std::vector<ProfilerStage> m_stages;
    std::unordered_map<std::string, int> m_stage_map;
    bool m_enabled;
    
    float m_total_frame_ms;
    float m_frame_history[PROFILER_HISTORY_SIZE];
    int m_frame_history_idx;
    int m_frame_count;
};

// ============================================================================
// SCOPED PROFILER - RAII helper for automatic start/stop
// ============================================================================

class ScopedProfiler {
public:
    ScopedProfiler(const std::string& name, cudaStream_t stream = 0) 
        : m_name(name), m_stream(stream) {
        Profiler::getInstance().startStage(name, stream);
    }
    
    ~ScopedProfiler() {
        Profiler::getInstance().stopStage(m_name, m_stream);
    }
    
private:
    std::string m_name;
    cudaStream_t m_stream;
};

// Macro for easy scoped profiling
#define PROFILE_SCOPE(name) ScopedProfiler _profiler_##__LINE__(name)
#define PROFILE_GPU_SCOPE(name) ScopedProfiler _profiler_##__LINE__(name, 0)

#endif // PROFILER_H
