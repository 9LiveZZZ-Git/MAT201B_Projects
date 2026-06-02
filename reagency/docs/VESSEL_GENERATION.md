# Generating the WoSW vessel (`vessel.wswv`)

The "vessel" is the morphing Gaussian-splat cloud at the core of the piece — `K` keyframes ×
`G` points, each `[pos3, rgb3, opacity, sigma]` as float16 (the `WSWV` format `VesselSplats`
loads). Each keyframe comes from one corpus image turned into a 3D object, then packed by
`factory/stage_d_vessel.py`.

Two image-to-3D routes produce the per-keyframe `.ply` files; both end at the same `stage_d`
pack. Pick one.

## Format reality (read this first)
Neither route's tool is a drop-in "image → gaussian splat." The vessel only needs **dense
colored 3D points**, so:
- **trellis-mac** outputs a **textured GLB mesh** → we sample its surface into a colored
  point-cloud `.ply` (the "splats" are mesh-surface-sampled points — dense, colored, high
  fidelity, but not native gaussians).
- **Replicate `firtoz/trellis`** can output a **native gaussian `.ply`** directly (falls back
  to GLB→points if a build only returns GLB).

Visually the vessel is the same dense colored morphing cloud either way.

## Spec mapping (vs the old vessel-colab "128 / G=30000 / STEPS=50")
| Colab spec | trellis-mac | Replicate (firtoz/trellis) |
|---|---|---|
| 128 splats | 128 corpus images → 128 `.ply` keyframes | same |
| `G=30000` gaussians/keyframe | sample **30 000 surface points** → `stage_d --G 30000` | native gaussians, `stage_d --G 30000` |
| `STEPS=50` | n/a → `--pipeline-type 1024_cascade` (highest) | `ss_sampling_steps` / `slat_sampling_steps` ≈ 25–50 |
| color fidelity | `--texture-size 2048` | `texture_size` |

`stage_d --max-mb 80` keeps 128×30000 (~61 MB) — all keyframes survive; pairs with `DENS=4`
in `VesselSplats.cpp`.

---

## Route A — trellis-mac (LOCAL, free, ~11 h, Apple-Silicon only)
CUDA-free MPS port of Microsoft TRELLIS.2 (<https://github.com/shivampkumar/trellis-mac>).
Driver: `factory/bake_vessel_trellis_mac.py`.

**One-time setup (you do this — it's a 15 GB download + Metal build + gated-model auth):**
1. HuggingFace: request access (instant approval) to
   `facebook/dinov3-vitl16-pretrain-lvd1689m` and `briaai/RMBG-2.0`.
2. Build it (Python 3.11+, **24 GB+ unified memory**, ~15 GB weights, Xcode Metal toolchain):
   ```bash
   git clone https://github.com/shivampkumar/trellis-mac && cd trellis-mac
   bash setup.sh                       # (or SKIP_METAL=1 bash setup.sh)
   .venv/bin/python generate.py some.jpg   # sanity check (~5 min)
   ```
3. In the python that runs the driver: `pip install trimesh numpy`.

**Run (from `allolib_playground`; ~5 min/image → ~11 h for 128; resumable — skips done `.ply`):**
```bash
# smoke test 2 images first (~10 min):
WSW_N=2 WSW_TRELLIS_DIR=/path/to/trellis-mac \
  python3 MAT201B_Projects/reagency/factory/bake_vessel_trellis_mac.py
# full run:
WSW_TRELLIS_DIR=/path/to/trellis-mac \
  python3 MAT201B_Projects/reagency/factory/bake_vessel_trellis_mac.py
```
Knobs (env): `WSW_N` (count), `WSW_PIPELINE` (`512|1024|1024_cascade`), `WSW_TEXSIZE`
(`512|1024|2048`), `WSW_GLB_PTS` (points/mesh, keep ≥ `--G`), `WSW_TRELLIS_DIR`,
`WSW_TRELLIS_PY` (venv python override).

---

## Route B — Replicate `firtoz/trellis` (HOSTED, ~$6–13, minutes, native gaussians)
Driver: `factory/bake_vessel_replicate.py`. Runs from the Mac; only the per-image inference is remote.
```bash
pip install replicate trimesh numpy
export REPLICATE_API_TOKEN=...        # replicate.com/account/api-tokens
# smoke test 4 images (~$0.40):
WSW_N=4 python3 MAT201B_Projects/reagency/factory/bake_vessel_replicate.py
# full run:
python3 MAT201B_Projects/reagency/factory/bake_vessel_replicate.py
```

---

## Pack (both routes end here)
```bash
cd MAT201B_Projects/reagency/factory
python3 stage_d_vessel.py --plys work/vessels --G 30000 --max-mb 80   # -> ../assets/vessel.wswv
```
Then `./run.sh MAT201B_Projects/reagency/src/main_reagency.cpp` loads the new vessel automatically.
(`vessel.wswv` is tracked via git-LFS — see `DOME_DEPLOY.md` — so after re-baking, commit it or
re-bundle for the dome.)

## Tradeoff
| | trellis-mac (A) | Replicate (B) |
|---|---|---|
| cost | free | ~$6–13 |
| time | ~11 h on your Mac | minutes (cloud A100) |
| setup | heavy one-time (15 GB, Metal, gated models) | a token |
| privacy | local | images uploaded to a 3rd party |
| output | mesh → sampled points | native gaussian `.ply` |
