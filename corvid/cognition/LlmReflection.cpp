#ifdef CORVID_USE_LLM
// Tier A per-agent reflection — llama.cpp batched decode with GBNF grammar.
// Threading: one background thread owns the llama_model + llama_context for
// the entire sim lifetime.  Sim thread only touches the job/result queues via
// submit() and drain() — both non-blocking.  Spec §2.5.2, §12.2 principle #3.
#include "LlmReflection.hpp"
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace corvid {

// ---------------------------------------------------------------------------
// GBNF grammar string — hard-coded so we never fail to find a file at runtime.
// Matches: [{"kind":"seek_food","weight":0.80},...]
// ---------------------------------------------------------------------------
// Grammar-constrained sampling (llama_sampler_init_grammar) throws on Gemma's `[` token.
// We use temperature sampling instead; the instruction-tuned model produces valid JSON,
// and parseGoals() falls back to heuristic on any parse failure.
// The canonical schema is documented in assets/goal.gbnf.

// ---------------------------------------------------------------------------
// Prompt builder (Gemma 4 chat template)
// ---------------------------------------------------------------------------
static std::string buildPrompt(const ReflectJob& job) {
    char buf[1024];
    float age = job.sim_time - job.birth_t;
    std::snprintf(buf, sizeof(buf),
        "<start_of_turn>user\n"
        "You are a corvid agent (id=%u). "
        "Energy: %.2f. Age: %.1fs. "
        "Recent events: %s\n"
        "Emit a JSON goal list with 1-%d goals. "
        "Each goal: {\"kind\": <seek_food|avoid_predator|explore|rest>, "
        "\"weight\": <0.00-1.00>}.\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n",
        job.agent_id, job.energy, age,
        job.digest[0] ? job.digest : "nothing notable",
        MAX_GOALS);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Token helpers (mirrors LlmSmoke.cpp)
// ---------------------------------------------------------------------------
static const llama_vocab* getVocab(llama_model* m) {
    return llama_model_get_vocab(m);
}

static std::vector<llama_token> tokenize(llama_model* m,
                                          const std::string& text,
                                          bool addSpecial) {
    const llama_vocab* v = getVocab(m);
    std::vector<llama_token> toks(text.size() + 32);
    int n = llama_tokenize(v, text.c_str(), int(text.size()),
                           toks.data(), int(toks.size()), addSpecial, false);
    if (n < 0) {
        toks.resize(-n);
        n = llama_tokenize(v, text.c_str(), int(text.size()),
                           toks.data(), int(toks.size()), addSpecial, false);
    }
    toks.resize(n > 0 ? n : 0);
    return toks;
}

static std::string tokStr(llama_model* m, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(getVocab(m), tok, buf, sizeof(buf), 0, false);
    return n > 0 ? std::string(buf, n) : std::string{};
}

static bool isEog(llama_model* m, llama_token tok) {
    return llama_vocab_is_eog(getVocab(m), tok);
}

// ---------------------------------------------------------------------------
// Parse GBNF-constrained output into GoalEntry array
// ---------------------------------------------------------------------------
static int parseGoals(const std::string& json, GoalEntry* out, int maxGoals) {
    int n = 0;
    const char* p = json.c_str();
    while (n < maxGoals && (p = std::strstr(p, "\"kind\""))) {
        p += 6;
        GoalEntry e;
        if      (std::strstr(p, "\"seek_food\""))       e.kind = GoalKind::SEEK_FOOD;
        else if (std::strstr(p, "\"avoid_predator\""))  e.kind = GoalKind::AVOID_PREDATOR;
        else if (std::strstr(p, "\"explore\""))         e.kind = GoalKind::EXPLORE;
        else if (std::strstr(p, "\"rest\""))             e.kind = GoalKind::REST;
        else { ++p; continue; }

        const char* wt = std::strstr(p, "\"weight\"");
        if (!wt) break;
        wt += 8;
        while (*wt && (*wt == ':' || *wt == ' ' || *wt == '\t')) ++wt;
        e.weight = std::strtof(wt, nullptr);
        e.weight = std::max(0.f, std::min(1.f, e.weight));
        out[n++] = e;
        p = wt;
    }
    return n;
}

// ---------------------------------------------------------------------------
// Run one batch (up to REFLECT_BATCH jobs) — called from thread only
// ---------------------------------------------------------------------------
static std::vector<ReflectResult> runBatch(const std::vector<ReflectJob>& jobs,
                                            llama_model* model,
                                            llama_context* ctx)
{
    const int nSeq = int(jobs.size());
    if (FILE* f = std::fopen("reflection.log", "a"))
        { std::fprintf(f, "runBatch: nSeq=%d\n", nSeq); std::fclose(f); }

    // Build prompts and tokenize
    std::vector<std::vector<llama_token>> seqToks(nSeq);
    int totalPrefill = 0;
    for (int s = 0; s < nSeq; ++s) {
        seqToks[s] = tokenize(model, buildPrompt(jobs[s]), /*addSpecial=*/true);
        totalPrefill += int(seqToks[s].size());
    }

    // Prefill batch — one token per sequence at a time, interleaved
    llama_batch prefill = llama_batch_init(totalPrefill, 0, nSeq);
    std::vector<int> promptLen(nSeq);
    for (int s = 0; s < nSeq; ++s) {
        const auto& toks = seqToks[s];
        for (int i = 0; i < int(toks.size()); ++i) {
            prefill.token   [prefill.n_tokens] = toks[i];
            prefill.pos     [prefill.n_tokens] = i;
            prefill.n_seq_id[prefill.n_tokens] = 1;
            prefill.seq_id  [prefill.n_tokens][0] = s;
            prefill.logits  [prefill.n_tokens] = (i == int(toks.size()) - 1) ? 1 : 0;
            ++prefill.n_tokens;
        }
        promptLen[s] = int(toks.size());
    }

    std::vector<ReflectResult> results(nSeq);
    for (int s = 0; s < nSeq; ++s) {
        results[s].slot     = jobs[s].slot;
        results[s].agent_id = jobs[s].agent_id;
        results[s].from_llm = true;
    }

    if (FILE* f = std::fopen("reflection.log", "a"))
        { std::fprintf(f, "runBatch: totalPrefill=%d\n", totalPrefill); std::fclose(f); }
    if (llama_decode(ctx, prefill)) {
        llama_batch_free(prefill);
        for (int s = 0; s < nSeq; ++s)
            results[s] = heuristicReflect(jobs[s]);
        return results;
    }
    llama_batch_free(prefill);

    // One grammar sampler per sequence (temp=0.7, grammar-constrained)
    std::vector<llama_sampler*> samplers(nSeq, nullptr);
    for (int s = 0; s < nSeq; ++s) {
        auto params = llama_sampler_chain_default_params();
        samplers[s] = llama_sampler_chain_init(params);
        llama_sampler_chain_add(samplers[s], llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(samplers[s], llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    // Track last-prefill batch positions for first-token sampling
    std::vector<int> lastPrefillBatchPos(nSeq);
    {
        int off = 0;
        for (int s = 0; s < nSeq; ++s) {
            lastPrefillBatchPos[s] = off + promptLen[s] - 1;
            off += promptLen[s];
        }
    }

    // First generated token per sequence
    std::vector<llama_token> lastTok(nSeq);
    std::vector<bool>        done(nSeq, false);
    std::vector<int>         pos(nSeq);
    std::vector<std::string> outputs(nSeq);
    for (int s = 0; s < nSeq; ++s) {
        pos[s] = promptLen[s];
        lastTok[s] = llama_sampler_sample(samplers[s], ctx, lastPrefillBatchPos[s]);
        if (isEog(model, lastTok[s])) done[s] = true;
        else outputs[s] += tokStr(model, lastTok[s]);
        llama_sampler_accept(samplers[s], lastTok[s]);
    }

    // Generation loop — one step batch per token position
    llama_batch step = llama_batch_init(nSeq, 0, nSeq);
    for (int t = 1; t < ReflectionThread::MAX_NEW_TOKENS; ++t) {
        step.n_tokens = 0;
        std::vector<int> active;
        for (int s = 0; s < nSeq; ++s) {
            if (done[s]) continue;
            step.token   [step.n_tokens] = lastTok[s];
            step.pos     [step.n_tokens] = pos[s]++;
            step.n_seq_id[step.n_tokens] = 1;
            step.seq_id  [step.n_tokens][0] = s;
            step.logits  [step.n_tokens] = 1;
            ++step.n_tokens;
            active.push_back(s);
        }
        if (active.empty()) break;
        if (llama_decode(ctx, step)) break;

        int si = 0;
        for (int s : active) {
            lastTok[s] = llama_sampler_sample(samplers[s], ctx, si++);
            if (isEog(model, lastTok[s])) { done[s] = true; continue; }
            outputs[s] += tokStr(model, lastTok[s]);
            llama_sampler_accept(samplers[s], lastTok[s]);
            if (int(outputs[s].size()) >= 200) done[s] = true;
        }
    }
    llama_batch_free(step);

    // Clear KV/memory cache so next batch starts fresh
    llama_memory_clear(llama_get_memory(ctx), true);

    // Free samplers
    for (auto* sm : samplers) if (sm) llama_sampler_free(sm);

    // Parse results
    for (int s = 0; s < nSeq; ++s) {
        int ng = parseGoals(outputs[s], results[s].goals, MAX_GOALS);
        if (ng == 0) {
            // Grammar-constrained output failed parse — use heuristic
            results[s] = heuristicReflect(jobs[s]);
        } else {
            results[s].n_goals = ng;
            std::strncpy(results[s].text, outputs[s].c_str(), 255);
            results[s].text[255] = '\0';
        }
    }

    return results;
}

// ---------------------------------------------------------------------------
// ReflectionThread public API
// ---------------------------------------------------------------------------

bool ReflectionThread::start(const std::string& model_path, int n_gpu_layers) {
    model_path_    = model_path;
    n_gpu_layers_  = n_gpu_layers;
    running_.store(true);
    thread_ = std::thread(&ReflectionThread::threadMain, this);
    // Wait briefly for model load result
    for (int i = 0; i < 50 && running_.load() && !model_loaded; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
  try {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = n_gpu_layers_;
    llama_model* model    = llama_model_load_from_file(model_path_.c_str(), mp);

    if (!model) {
        std::snprintf(status_msg, sizeof(status_msg),
                      "FAILED to load: %s", model_path_.c_str());
        if (FILE* f = std::fopen("reflection.log", "a"))
            { std::fprintf(f, "[ReflectionThread] %s\n", status_msg); std::fclose(f); }
        running_.store(false);
        llama_backend_free();
        return;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = N_CTX * REFLECT_BATCH;  // total KV context (seqs × tokens per seq)
    cp.n_batch   = N_CTX * REFLECT_BATCH;  // must be >= max prefill tokens across the batch
    cp.n_seq_max = REFLECT_BATCH;

    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::snprintf(status_msg, sizeof(status_msg), "FAILED to create context");
        llama_model_free(model);
        running_.store(false);
        llama_backend_free();
        return;
    }

    model_loaded = true;
    std::snprintf(status_msg, sizeof(status_msg), "ready (ngl=%d)", n_gpu_layers_);
    if (FILE* f = std::fopen("reflection.log", "a"))
        { std::fprintf(f, "[ReflectionThread] model loaded, %s\n", status_msg); std::fclose(f); }

    while (running_.load()) {
        // Collect up to REFLECT_BATCH jobs
        std::vector<ReflectJob> batch;
        {
            std::unique_lock<std::mutex> lk(job_mtx_);
            job_cv_.wait_for(lk, std::chrono::milliseconds(100),
                             [&]{ return !job_queue_.empty() || !running_.load(); });
            while (!job_queue_.empty() && int(batch.size()) < REFLECT_BATCH) {
                batch.push_back(job_queue_.front());
                job_queue_.pop();
            }
        }
        if (batch.empty()) continue;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto results = runBatch(batch, model, ctx);
        auto t1 = std::chrono::high_resolution_clock::now();
        last_batch_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        ++batches_done;

        if (FILE* f = std::fopen("reflection.log", "a")) {
            std::fprintf(f, "batch %d: %d agents, %.0f ms\n",
                         batches_done, int(batch.size()), last_batch_ms);
            for (auto& r : results)
                if (r.from_llm)
                    std::fprintf(f, "  slot=%d: %s\n", r.slot, r.text);
            std::fclose(f);
        }

        {
            std::lock_guard<std::mutex> lk(result_mtx_);
            for (auto& r : results) result_queue_.push(r);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::snprintf(status_msg, sizeof(status_msg), "stopped");
  } catch (const std::exception& e) {
    if (FILE* f = std::fopen("reflection.log", "a"))
        { std::fprintf(f, "[ReflectionThread] EXCEPTION: %s\n", e.what()); std::fclose(f); }
    running_.store(false);
    model_loaded = false;
  } catch (...) {
    if (FILE* f = std::fopen("reflection.log", "a"))
        { std::fprintf(f, "[ReflectionThread] UNKNOWN EXCEPTION\n"); std::fclose(f); }
    running_.store(false);
    model_loaded = false;
  }
}

} // namespace corvid
#endif // CORVID_USE_LLM
