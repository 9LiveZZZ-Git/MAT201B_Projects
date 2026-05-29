# Vessel pipeline — image-to-3D Gaussian splats → `vessel.wswv` (Stage D)

The outer "vessel" is the machine's **image-to-3D hallucination of each cluster's
representative corpus photo**, melted between clusters at runtime. Generation is OFFLINE
(on Google Colab Pro+, A100 ~40 GB); the allolib runtime only loads a compact binary — no
runtime ML. This doc is the verified runbook (from the vessel-design workflow).

> **Easiest path: open `factory/wosw_colab.ipynb` in Colab and Run All** — it runs this whole
> pipeline (corpus → embeddings → galaxy + webs → vessels → `vessel.wswv`) and pushes the assets
> back. `prep_vessel_inputs.py --mode all` makes **one vessel per image**, not just per cluster.
> The steps below are the manual reference.

## The one decision that makes or breaks it: correspondence
Two **independently generated** splat clouds have **no point correspondence** — so a naïve
fixed-index lerp makes every point fly along an arbitrary chord = global *swimming*, not a
morph, and an additive smoke-melt only hides it while points are dispersed (it fails exactly
at the legible endpoints). The fix is to impose a **canonical ordering offline** so index `i`
means "the same place" in every keyframe:

- **Default (what `stage_d_vessel.py` does):** prune each keyframe to a fixed `G`, normalize
  all keyframes to ONE global AABB, then **sort every keyframe by a Morton (Z-order) code**
  over that shared space. Index-by-index lerp is then locally coherent → a real morph, with
  zero runtime cost and numpy-only deps.
- **Higher quality (optional, later):** GaussianCube-style fixed 32³ grid + Optimal-Transport
  assignment (`scipy.optimize.linear_sum_assignment`). Swap in if Morton isn't clean enough.

`G` (Gaussians per keyframe) **must be identical across all keyframes** — enforced by Stage D.

## Step 1 — generate one `.ply` per cluster (Google Colab Pro+, A100 ~40 GB)

Generation now runs on **Colab Pro+ (A100, ~40 GB)** — VRAM is no longer a constraint, so we run
the best models at full quality. (OFFLINE only; the runtime/dome budget is unchanged.) You only
need **one hero per cluster** (~6–24, not 704); `stage_b` writes `work/cluster_reps.json` with the
representative corpus image per cluster.

Get the corpus onto Colab (images are gitignored) — clone, then re-fetch from the ledger:
```bash
!git clone -b world-of-shadow-work https://github.com/9LiveZZZ-Git/MAT201B_Projects.git
%cd MAT201B_Projects/reagency/factory && !python3 download_from_csv.py   # re-fetch corpus images
```

### Recommended: LGM (simple, fast, reliable) — trivial on the A100
Repo: https://github.com/3DTopia/LGM (ECCV 2024). Feed-forward, a few seconds/image on an A100.
Emits 14-channel 3DGS `.ply` (pos, f_dc×3, opacity, scale×3, rot×4 — no normals, no f_rest).
```bash
# Colab Pro+ A100: torch + CUDA are preinstalled; just add the extras
pip install -U xformers
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization && pip install ./diff-gaussian-rasterization
pip install git+https://github.com/NVlabs/nvdiffrast
git clone https://github.com/3DTopia/LGM && cd LGM && pip install -r requirements.txt
# download pretrained/model_fp16.safetensors per the repo, then:
python infer.py big --resume pretrained/model_fp16.safetensors \
  --workspace .../reagency/factory/work/vessels \
  --test_path <a cluster-rep image OR a dir of them>     # LGM rembg's the background itself
```
### Top quality (now that VRAM allows): TRELLIS
Repo: https://github.com/microsoft/TRELLIS — Microsoft's image-to-3D structured-latent model,
the best Gaussian quality but heavier (~16–24 GB; comfortable on the 40 GB A100). Follow its
README for the CUDA-extension-heavy install, run its image-to-3D pipeline, and export the 3DGS
`.ply`. Use it for hero vessels you want cleaner; LGM for fast breadth — Stage D auto-detects the
`.ply` schema either way. (The old 8 GB fallbacks — DreamGaussian / TriplaneGaussian — are no
longer needed with the A100.)

### Glitch aesthetic = free
Single-image 3D models hallucinate the unseen back (Janus artifacts, floaters, impossible
thickness) — exactly the "machine dreaming" look. Do **not** clean/prune the raw `.ply`; for
extra divergence, generate 2–3 seeds per cluster as additional morph targets.

## Step 2 — pack to `vessel.wswv`
```bash
python3 stage_d_vessel.py --plys factory/work/vessels --G 12000
# -> ../assets/vessel.wswv
```
`stage_d_vessel.py` auto-detects LGM (14-field) vs Inria (`f_rest_*`) schemas, inverts the
activations (rgb = 0.5 + 0.2820948·f_dc; opacity = sigmoid; scale = exp), recenters +
globally normalizes, imposes the Morton canonical order, and packs fp16.

### `WSWV` format (mirrors the WSWP/WSWE family)
```
[magic "WSWV"][i32 version=1][i32 K keyframes][i32 G gaussians/keyframe][i32 stride=8]
[6×f32 global AABB: min.xyz, max.xyz]
[K × G × 8 float16]   per Gaussian: pos.xyz(3) + rgb(3) + opacity(1) + sigma(1)
```
Quaternions are dropped — the renderer is isotropic additive point-sprites, so rotation isn't
used. `G=12000, K=12` ≈ **2.3 MB**. fp16 is a disk-only choice (allolib has no fp16 vertex
path; the loader expands to f32). Positions are normalized to ~[-1,1], so fp16 is precise.

## Step 3 — ship it
```bash
cd reagency && git add assets/vessel.wswv && git commit -m "assets: vessel bank" && git push
```
The runtime `VesselSplats` module (being written next) loads this, keeps only the two active
keyframes alive, lerps them per frame in the vertex shader, applies the smoky melt
(tent scale-up/opacity-down + curl-noise), and crossfades with the galaxy by `depth`.

## Budget / caveats (verified)
- **Dome fill-rate:** additive glow sprites have no early-Z. Keep `G ≈ 8k–16k`, cap on-screen
  point size (~≤24 px), and only ever blend **two** keyframes — never stack all K.
- **Image res:** your corpus thumbs (414–900 px) are above the 256 px both models consume, so
  no re-fetch needed; optionally re-fetch the ~10–24 cluster heroes at ≥512 px for cleaner
  geometry.
- **Coordinate frame:** Stage D recenters + unit-normalizes; if a vessel looks mis-oriented vs
  the galaxy, check the generator's up-axis on the first asset before batching.
