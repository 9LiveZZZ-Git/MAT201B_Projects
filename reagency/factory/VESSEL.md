# Vessel pipeline — image-to-3D Gaussian splats → `vessel.wswv` (Stage D)

The outer "vessel" is the machine's **image-to-3D hallucination of each cluster's
representative corpus photo**, melted between clusters at runtime. Generation is OFFLINE
(on the 4060 Ti); the allolib runtime only loads a compact binary and morphs it — no
runtime ML. This doc is the verified runbook (from the vessel-design workflow).

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

## Step 1 — generate one `.ply` per cluster (on the 4060 Ti)

You only need **one hero per cluster** (clustering yields ~6–24 clusters, not 704). After
`make_assets.sh`, `stage_b` writes `work/cluster_reps.json` listing the representative corpus
image per cluster — feed those to the generator.

### Primary: LGM (best quality; fits the 16 GB 4060 Ti)
Repo: https://github.com/3DTopia/LGM (ECCV 2024). Feed-forward, ~10–20 s/image on a 4060 Ti.
Emits 14-channel 3DGS `.ply` (pos, f_dc×3, opacity, scale×3, rot×4 — no normals, no f_rest).
```bash
# env (cu118 supports the Ada sm_89 4060 Ti cleanly)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization && pip install ./diff-gaussian-rasterization
pip install git+https://github.com/NVlabs/nvdiffrast
git clone https://github.com/3DTopia/LGM && cd LGM && pip install -r requirements.txt
# download pretrained/model_fp16.safetensors per the repo, then:
python infer.py big --resume pretrained/model_fp16.safetensors \
  --workspace .../reagency/factory/work/vessels \
  --test_path <a cluster-rep image OR a dir of them>     # LGM rembg's the background itself
```
**8 GB 4060 Ti:** LGM's ~10 GB peak (ImageDream + U-Net resident) will OOM. Either offload
between stages (`del pipe; torch.cuda.empty_cache()` after the 4-view diffusion, then load the
U-Net), rent a 24 GB cloud GPU for the one-time batch, or use the fallback below.

### Fallback: DreamGaussian (confirmed on 8 GB)
Repo: https://github.com/dreamgaussian/dreamgaussian (~2 min/image on an 8 GB 3070). Needs an
RGBA cutout (`rembg`); produces a 3DGS `.ply` you pass to Stage D the same way.

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
