#pragma once
// ReflectionThread — Tier A per-agent reflection via Gemma 4 E2B (spec §2.5.2).
// Only compiled when CORVID_USE_LLM is defined.
#ifdef CORVID_USE_LLM

#include "Reflection.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace corvid {

struct ReflectionThread {
    static constexpr int REFLECT_BATCH  = 32;   // n_seq_max (spec §3.3)
    static constexpr int MAX_NEW_TOKENS = 96;   // spec §3.3
    static constexpr int N_CTX          = 2048; // context per sequence
    static constexpr int JOB_QUEUE_CAP  = 512;

    // Start background thread; returns false if model fails to load.
    bool start(const std::string& model_path, int n_gpu_layers = 0);  // args kept for API parity; ignored
    void stop();
    ~ReflectionThread() { stop(); }

    // Non-blocking submit. Drops jobs if queue is at capacity.
    void submit(const ReflectJob* jobs, int n);

    // Non-blocking drain. Returns number of results written to out[].
    int drain(ReflectResult* out, int max_out);

    float last_batch_ms   = 0.f;
    int   batches_done    = 0;
    bool  model_loaded    = false;
    char  status_msg[128] = "not started";

private:
    std::thread             thread_;
    std::atomic<bool>       running_{false};

    std::mutex              job_mtx_;
    std::condition_variable job_cv_;
    std::queue<ReflectJob>  job_queue_;

    std::mutex              result_mtx_;
    std::queue<ReflectResult> result_queue_;

    std::string model_path_;
    int         n_gpu_layers_ = 99;

    void threadMain();
};

} // namespace corvid
#endif // CORVID_USE_LLM
