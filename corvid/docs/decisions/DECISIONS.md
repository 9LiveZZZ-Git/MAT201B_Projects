# Corvid — Architecture Decisions

## LLM inference: CPU-only for both tiers

Both Tier A (Gemma 4 E2B) and Tier B (Gemma 4 31B) run on the host CPU with
`n_gpu_layers=0`. The machine has 128 GB DDR5 and an RTX 4060 Ti with 8 GB VRAM.
Running inference on CPU avoids contention with the allolib GPU render loop and
keeps the full VRAM budget (~5 GB) for graphics. 31B Q4_K_M fits comfortably in
RAM (~18 GB). Decision made week 1.

## llama.cpp tag: b5570

Pinned to avoid API churn in a fast-moving repo. Update only when a required
feature (multi-seq batch, speculative decoding) lands. See `pins.toml`.

## DistributedAppWithState state blob: 5.6 KB POD

`SimpleSharedState` packs 200 positions (float[3]) + 200 quaternions (float[4])
plus two uint32 counters. Total ~5.6 KB, well under the 20 KB UDP MTU budget.
No compression needed at week 1 scale.

## Audio trigger rate-limiting: 100 ms per agent

PolySynth voice triggers are rate-limited to one per agent per 100 ms to prevent
voice starvation and audio thread xruns. Voice pool pre-allocated to 32 in
`onCreate()` so no heap allocation occurs on the audio thread.

## Toroidal world: W=10, spatial hash cell = neighbor radius T

Cell size equals the neighbor radius so a 3³ probe covers all possible neighbors
exactly once. Toroidal cell indices wrapped via `((cx % dim) + dim) % dim`.
