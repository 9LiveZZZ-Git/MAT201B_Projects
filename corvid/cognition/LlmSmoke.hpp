#pragma once
#include <string>
#include <vector>

namespace corvid {

struct LlmConfig {
  std::string modelPath;
  int         nCtx       = 8192;
  int         nBatch     = 512;
  int         nSeqMax    = 32;
  int         nGpuLayers = 0;   // CPU-only
};

// Single-sequence inference. Returns decoded text or empty string on error.
std::string llmInferSingle(const LlmConfig& cfg,
                           const std::string& prompt,
                           int maxTokens = 128);

// Multi-sequence batch inference.
// prompts.size() capped at cfg.nSeqMax.
// Returns one decoded string per prompt.
std::vector<std::string> llmInferBatch(const LlmConfig& cfg,
                                       const std::vector<std::string>& prompts,
                                       int maxTokens = 128);

} // namespace corvid
