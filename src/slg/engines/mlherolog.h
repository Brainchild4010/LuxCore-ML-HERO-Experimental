#ifndef SLG_MLHERO_LOG15GB_H
#define SLG_MLHERO_LOG15GB_H

// ML LuxCore HERO 15gb: one run-specific filename for the existing CPU diagnostics.
// The diagnostics retain their existing mutexes and FILE* write semantics.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <sstream>
#include <iomanip>

namespace slg {
namespace mlhero15gb {
inline std::mutex &PathMutex() { static std::mutex m; return m; }
inline std::chrono::steady_clock::time_point &StartTime() { static auto t = std::chrono::steady_clock::now(); return t; }
inline std::string &ActivePath() { static std::string p; return p; }
inline const char *FilePath() { return ActivePath().c_str(); }
inline bool Begin(const char *engine, const bool hero, const char *build, const char *phase) {
    std::lock_guard<std::mutex> guard(PathMutex());
    namespace fs = std::filesystem;
    const fs::path dir("C:\\LuxCoreCustom");
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return false;
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tmValue{};
#ifdef _WIN32
    localtime_s(&tmValue, &t);
#else
    localtime_r(&t, &tmValue);
#endif
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;
    static std::atomic<unsigned long long> runSerial{0};
    const auto serial = ++runSerial;
    std::ostringstream name;
    name << "ML_" << build << '_' << phase << '_' << engine << '_'
         << (hero ? "ON" : "OFF") << '_' << std::put_time(&tmValue, "%Y%m%d_%H%M%S")
         << '_' << std::setw(6) << std::setfill('0') << micros << '_' << serial << ".log";
    const fs::path fullPath = dir / name.str();
    ActivePath() = fullPath.string();
    StartTime() = std::chrono::steady_clock::now();
    std::fprintf(stderr, "ML HERO 15gb log: %s\n", ActivePath().c_str());
    return true;
}
// Called after the CPU engine has joined its worker threads. STOPPED is
// intentionally neutral: the engine cannot distinguish a user cancellation
// from a render that reached its halt condition at this callback.
inline void Finish(const char *engine) {
    std::lock_guard<std::mutex> guard(PathMutex());
    if (ActivePath().empty()) return;
    FILE *f = std::fopen(ActivePath().c_str(), "a");
    if (f) {
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - StartTime()).count();
        std::fprintf(f, "\nML HERO RENDER_END engine=%s status=STOPPED elapsedSeconds=%.3f statusDetail=halt_reason_not_available\n", engine, seconds);
        std::fclose(f);
    }
    ActivePath().clear();
}
}
}
#endif
