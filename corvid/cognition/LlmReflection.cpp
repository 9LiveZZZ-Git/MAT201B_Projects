#ifdef CORVID_USE_LLM
// ReflectionThread — background per-agent reflection.
//
// allolib-native rewrite: the LLM (llama.cpp / Gemma) has been removed. There is
// no native LLM, so reflection runs the procedural reflector (Reflection.cpp) on
// a background thread, preserving the exact public API (start/stop/submit/drain)
// so main_corvid.cpp's scheduler and HUD are unchanged. The CORVID_USE_LLM flag
// now simply means "threaded reflection enabled" — no model is loaded.
#include "LlmReflection.hpp"
#include <chrono>
#include <cstdio>
#include <vector>

namespace corvid {

bool ReflectionThread::start(const std::string& model_path, int n_gpu_layers) {
    model_path_   = model_path;       // ignored (no model); kept for API parity
    n_gpu_layers_ = n_gpu_layers;
    running_.store(true);
    thread_ = std::thread(&ReflectionThread::threadMain, this);
    // Procedural reflector is always ready; wait briefly for the flag.
    for (int i = 0; i < 50 && running_.load() && !model_loaded; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    return model_loaded;
}

void ReflectionThread::stop() {
    if (!running_.exchange(false)) return;
    job_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void ReflectionThread::submit(const ReflectJob* jobs, int n) {
    std::lock_guard<std::mutex> lk(job_mtx_);
    for (int i = 0; i < n; ++i) {
        if (int(job_queue_.size()) >= JOB_QUEUE_CAP) break;
        job_queue_.push(jobs[i]);
    }
    job_cv_.notify_one();
}

int ReflectionThread::drain(ReflectResult* out, int max_out) {
    std::lock_guard<std::mutex> lk(result_mtx_);
    int n = 0;
    while (n < max_out && !result_queue_.empty()) {
        out[n++] = result_queue_.front();
        result_queue_.pop();
    }
    return n;
}

void ReflectionThread::threadMain() {
    model_loaded = true;
    std::snprintf(status_msg, sizeof(status_msg), "procedural reflector ready");

    while (running_.load()) {
        std::vector<ReflectJob> batch;
        {
            std::unique_lock<std::mutex> lk(job_mtx_);
            job_cv_.wait_for(lk, std::chrono::milliseconds(50),
                             [&] { return !job_queue_.empty() || !running_.load(); });
            while (!job_queue_.empty() && int(batch.size()) < REFLECT_BATCH) {
                batch.push_back(job_queue_.front());
                job_queue_.pop();
            }
        }
        if (batch.empty()) continue;

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<ReflectResult> results;
        results.reserve(batch.size());
        for (const auto& j : batch) results.push_back(heuristicReflect(j));
        auto t1 = std::chrono::high_resolution_clock::now();

        last_batch_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        ++batches_done;

        std::lock_guard<std::mutex> lk(result_mtx_);
        for (const auto& r : results) result_queue_.push(r);
    }
    std::snprintf(status_msg, sizeof(status_msg), "stopped");
}

} // namespace corvid
#endif // CORVID_USE_LLM
