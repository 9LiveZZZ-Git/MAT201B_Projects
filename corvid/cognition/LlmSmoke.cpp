// Thin wrapper over the llama.cpp C API for Corvid week-1 smoke test.
// Updated for llama.cpp b5575 API (vocab-based functions).

#include "LlmSmoke.hpp"
#include "llama.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace corvid {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const llama_vocab* getVocab(const llama_model* model) {
  return llama_model_get_vocab(model);
}

static std::vector<llama_token> tokenize(llama_model* model,
                                         const std::string& text,
                                         bool addSpecial) {
  const llama_vocab* vocab = getVocab(model);
  std::vector<llama_token> tokens(text.size() + 32);
  int count = llama_tokenize(vocab, text.c_str(), int(text.size()),
                             tokens.data(), int(tokens.size()),
                             addSpecial, false);
  if (count < 0) {
    tokens.resize(-count);
    count = llama_tokenize(vocab, text.c_str(), int(text.size()),
                           tokens.data(), int(tokens.size()),
                           addSpecial, false);
  }
  tokens.resize(count);
  return tokens;
}

static std::string tokenToStr(llama_model* model, llama_token tok) {
  char buf[256];
  int n = llama_token_to_piece(getVocab(model), tok, buf, sizeof(buf), 0, false);
  if (n < 0) return "";
  return std::string(buf, n);
}

// batchIdx: the token's position in the most recent batch (used as key into output_ids)
static llama_token sampleGreedy(llama_context* ctx, int batchIdx) {
  float*   logits = llama_get_logits_ith(ctx, batchIdx);
  int32_t  vocab  = llama_vocab_n_tokens(getVocab(llama_get_model(ctx)));
  return llama_token(std::max_element(logits, logits + vocab) - logits);
}

static bool isEog(llama_model* model, llama_token tok) {
  return llama_vocab_is_eog(getVocab(model), tok);
}

// ---------------------------------------------------------------------------
// Load helpers — shared across both API calls
// ---------------------------------------------------------------------------

struct LlmHandle {
  llama_model*   model = nullptr;
  llama_context* ctx   = nullptr;

  LlmHandle(const LlmConfig& cfg) {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = cfg.nGpuLayers;

    model = llama_model_load_from_file(cfg.modelPath.c_str(), mp);
    if (!model) throw std::runtime_error("Failed to load model: " + cfg.modelPath);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = cfg.nCtx;
    cp.n_batch   = cfg.nBatch;
    cp.n_seq_max = cfg.nSeqMax;

    ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("Failed to create context");
  }

  ~LlmHandle() {
    if (ctx)   llama_free(ctx);
    if (model) llama_model_free(model);
    llama_backend_free();
  }
};

// ---------------------------------------------------------------------------
// Single-sequence inference
// ---------------------------------------------------------------------------

std::string llmInferSingle(const LlmConfig& cfg,
                           const std::string& prompt,
                           int maxTokens) {
  LlmHandle h(cfg);
  auto tokens = tokenize(h.model, prompt, /*addSpecial=*/true);

  llama_batch batch = llama_batch_init(int(tokens.size()), 0, 1);
  for (int i = 0; i < int(tokens.size()); ++i) {
    batch.token   [batch.n_tokens] = tokens[i];
    batch.pos     [batch.n_tokens] = i;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id  [batch.n_tokens][0] = 0;
    batch.logits  [batch.n_tokens] = (i == int(tokens.size()) - 1) ? 1 : 0;
    ++batch.n_tokens;
  }

  if (llama_decode(h.ctx, batch)) {
    llama_batch_free(batch);
    return "";
  }

  std::string output;
  int pos = int(tokens.size());

  // First token: sample from the last prefill position in the batch
  llama_token firstTok = sampleGreedy(h.ctx, int(tokens.size()) - 1);
  if (isEog(h.model, firstTok)) { llama_batch_free(batch); return output; }
  output += tokenToStr(h.model, firstTok);

  {
    llama_batch next = llama_batch_init(1, 0, 1);
    next.token[0] = firstTok; next.pos[0] = pos++; next.n_seq_id[0] = 1;
    next.seq_id[0][0] = 0; next.logits[0] = 1; next.n_tokens = 1;
    if (llama_decode(h.ctx, next)) { llama_batch_free(next); llama_batch_free(batch); return output; }
    llama_batch_free(next);
  }

  for (int t = 1; t < maxTokens; ++t) {
    llama_token tok = sampleGreedy(h.ctx, 0);  // step batch always has 1 token at idx 0
    if (isEog(h.model, tok)) break;

    output += tokenToStr(h.model, tok);

    llama_batch next = llama_batch_init(1, 0, 1);
    next.token   [0] = tok;
    next.pos     [0] = pos++;
    next.n_seq_id[0] = 1;
    next.seq_id  [0][0] = 0;
    next.logits  [0] = 1;
    next.n_tokens = 1;
    if (llama_decode(h.ctx, next)) { llama_batch_free(next); break; }
    llama_batch_free(next);
  }

  llama_batch_free(batch);
  return output;
}

// ---------------------------------------------------------------------------
// Batch inference (up to nSeqMax sequences, one llama_decode per token step)
// ---------------------------------------------------------------------------

std::vector<std::string> llmInferBatch(const LlmConfig& cfg,
                                       const std::vector<std::string>& prompts,
                                       int maxTokens) {
  LlmHandle h(cfg);
  const int nSeq = int(std::min(int(prompts.size()), cfg.nSeqMax));

  std::vector<std::vector<llama_token>> seqTokens(nSeq);
  for (int s = 0; s < nSeq; ++s)
    seqTokens[s] = tokenize(h.model, prompts[s], true);

  int totalTokens = 0;
  for (int s = 0; s < nSeq; ++s) totalTokens += int(seqTokens[s].size());

  llama_batch batch = llama_batch_init(totalTokens, 0, nSeq);
  std::vector<int> promptLen(nSeq);
  for (int s = 0; s < nSeq; ++s) {
    const auto& toks = seqTokens[s];
    for (int i = 0; i < int(toks.size()); ++i) {
      batch.token   [batch.n_tokens] = toks[i];
      batch.pos     [batch.n_tokens] = i;
      batch.n_seq_id[batch.n_tokens] = 1;
      batch.seq_id  [batch.n_tokens][0] = s;
      batch.logits  [batch.n_tokens] = (i == int(toks.size()) - 1) ? 1 : 0;
      ++batch.n_tokens;
    }
    promptLen[s] = int(toks.size());
  }

  if (llama_decode(h.ctx, batch)) {
    llama_batch_free(batch);
    return std::vector<std::string>(nSeq);
  }
  llama_batch_free(batch);

  std::vector<std::string> outputs(nSeq);
  std::vector<int>         pos(nSeq);
  std::vector<bool>        done(nSeq, false);
  for (int s = 0; s < nSeq; ++s) pos[s] = promptLen[s];

  // Track the batch index of each seq's last prompt token (where logits were enabled)
  std::vector<int> lastPrefillPos(nSeq);
  {
    int batchOff = 0;
    for (int s = 0; s < nSeq; ++s) {
      lastPrefillPos[s] = batchOff + promptLen[s] - 1;
      batchOff += promptLen[s];
    }
  }

  std::vector<llama_token> lastTok(nSeq);
  for (int s = 0; s < nSeq; ++s) {
    lastTok[s] = sampleGreedy(h.ctx, lastPrefillPos[s]);
    if (isEog(h.model, lastTok[s])) done[s] = true;
    else outputs[s] += tokenToStr(h.model, lastTok[s]);
  }

  llama_batch step = llama_batch_init(nSeq, 0, nSeq);
  for (int t = 1; t < maxTokens; ++t) {
    bool anyActive = false;
    step.n_tokens = 0;
    for (int s = 0; s < nSeq; ++s) {
      if (done[s]) continue;
      anyActive = true;
      step.token   [step.n_tokens] = lastTok[s];
      step.pos     [step.n_tokens] = pos[s]++;
      step.n_seq_id[step.n_tokens] = 1;
      step.seq_id  [step.n_tokens][0] = s;
      step.logits  [step.n_tokens] = 1;
      ++step.n_tokens;
    }
    if (!anyActive) break;
    if (llama_decode(h.ctx, step)) break;

    // In the step batch, active seqs occupy positions 0..step.n_tokens-1 in order.
    int stepPos = 0;
    for (int s = 0; s < nSeq; ++s) {
      if (done[s]) continue;
      lastTok[s] = sampleGreedy(h.ctx, stepPos++);
      if (isEog(h.model, lastTok[s])) done[s] = true;
      else outputs[s] += tokenToStr(h.model, lastTok[s]);
    }
  }

  llama_batch_free(step);
  return outputs;
}

} // namespace corvid
