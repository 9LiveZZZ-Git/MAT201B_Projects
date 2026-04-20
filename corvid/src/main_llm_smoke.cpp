// corvid_llm_smoke — Week 1 LLM plumbing verification.
// Usage:
//   corvid_llm_smoke --model PATH [--prompt TEXT] [--batch N] [--max_tokens N]

#include "cognition/LlmSmoke.hpp"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static void usage(const char* prog) {
  std::cerr
    << "Usage: " << prog << "\n"
    << "  --model PATH         path to .gguf file (required)\n"
    << "  --prompt TEXT        prompt string (default: built-in)\n"
    << "  --batch N            number of parallel sequences (1 or 2-32)\n"
    << "  --max_tokens N       max tokens to generate (default 128)\n";
}

int main(int argc, char** argv) {
  corvid::LlmConfig cfg;
  std::string       prompt =
    "You are a reflection engine for a bird simulation. "
    "Emit a 30-word description of curiosity.";
  int batch     = 1;
  int maxTokens = 128;

  for (int i = 1; i < argc; ++i) {
    if      (!strcmp(argv[i], "--model")      && i+1 < argc) cfg.modelPath = argv[++i];
    else if (!strcmp(argv[i], "--prompt")     && i+1 < argc) prompt        = argv[++i];
    else if (!strcmp(argv[i], "--batch")      && i+1 < argc) batch         = std::stoi(argv[++i]);
    else if (!strcmp(argv[i], "--max_tokens") && i+1 < argc) maxTokens     = std::stoi(argv[++i]);
    else { usage(argv[0]); return 1; }
  }

  if (cfg.modelPath.empty()) { usage(argv[0]); return 1; }

  batch = std::max(1, std::min(batch, cfg.nSeqMax));

  if (batch == 1) {
    std::cout << "[single] prompt: " << prompt << "\n\n";
    std::string out = corvid::llmInferSingle(cfg, prompt, maxTokens);
    if (out.empty()) {
      std::cerr << "ERROR: no output produced\n";
      return 1;
    }
    std::cout << out << "\n";
    std::cout << "\n[tokens generated: " << out.size() << " chars]\n";
  } else {
    // Build batch prompts: vary the last word per seq_id
    const char* concepts[] = {
      "curiosity",  "hunger",    "fear",      "joy",
      "loneliness", "wonder",    "playfulness","urgency",
      "calm",       "alertness", "confusion",  "determination",
      "grief",      "elation",   "boredom",    "anticipation",
      "trust",      "surprise",  "disgust",    "acceptance",
      "vigilance",  "ecstasy",   "admiration", "terror",
      "amazement",  "grief",     "interest",   "serenity",
      "pensiveness","apprehension","distraction","love"
    };
    std::vector<std::string> prompts;
    const char* base =
      "You are a reflection engine for a bird simulation. "
      "Emit a 30-word description of ";
    for (int s = 0; s < batch; ++s) {
      prompts.push_back(std::string(base) + concepts[s % 32] + ".");
    }

    std::cout << "[batch=" << batch << "] generating...\n\n";
    auto outputs = corvid::llmInferBatch(cfg, prompts, maxTokens);

    for (int s = 0; s < int(outputs.size()); ++s) {
      std::cout << "--- seq " << s << " [" << concepts[s % 32] << "] ---\n";
      std::cout << outputs[s] << "\n\n";
    }
    std::cout << "[batch complete: " << outputs.size() << " sequences]\n";
  }

  return 0;
}
