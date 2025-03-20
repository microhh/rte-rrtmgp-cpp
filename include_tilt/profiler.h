#ifndef PROFILER_H
#define PROFILER_H

#include <unordered_map>
#include <string>
#include <chrono>
#include <iostream>
#include <mutex>
#include <memory>

// Thread-safe profiler singleton class
class Profiler {
private:
    std::unordered_map<std::string, double> total_times;
    std::unordered_map<std::string, int> call_counts;
    std::mutex mutex;
    
    // Private constructor for singleton
    Profiler() = default;

public:
    // Delete copy/move operations
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    Profiler(Profiler&&) = delete;
    Profiler& operator=(Profiler&&) = delete;

    // Singleton access method
    static Profiler& instance() {
        static Profiler instance;
        return instance;
    }
    
    void add_measurement(const std::string& name, double time_ms) {
        std::lock_guard<std::mutex> lock(mutex);
        total_times[name] += time_ms;
        call_counts[name]++;
    }
    
    void report_and_reset() {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto& entry : total_times) {
            const std::string& name = entry.first;
            double avg_time = total_times[name] / call_counts[name];
            std::cout << name << " took average " << avg_time << " ms over " 
                      << call_counts[name] << " calls\n";
        }
        
        // Reset after reporting
        total_times.clear();
        call_counts.clear();
    }
};

// Redefine the macro to use the profiler singleton
#define PROFILE_BLOCK(name, code) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code; \
        auto end = std::chrono::high_resolution_clock::now(); \
        Profiler::instance().add_measurement(name, \
            std::chrono::duration<double, std::milli>(end - start).count()); \
    } while (0)

#endif // PROFILER_H
