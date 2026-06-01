# Deploying *World of Shadow Work* to the AlloSphere

How to take this repo from a fresh `git clone` to a synchronized multi-machine dome show.
Derived from a full portability audit of `src/main_reagency.cpp` + every viz/audio loader.

## TL;DR

A clone **builds and launches** anywhere (no hardcoded paths, no macOS-only runtime code).
But two things are NOT in git and must be handled per-deployment:

1. **Three large runtime assets** are gitignored (regenerable / over GitHub's size limits).
   Without them the piece still runs but loses the splat vessel, the "watch it think"
   emergence, and the recorded ghost voices. Ship them with `factory/make_dome_bundle.sh`.
2. **`distributed_app.toml`** (node roles) is gitignored on purpose. Author it from
   `distributed_app.toml.example`. Without a valid one, dome nodes go black.

Everything degrades gracefully — a missing asset disables its feature with a stderr warning,
never a crash — so you can stage the deploy and add fidelity incrementally.

## Asset inventory

| In a fresh clone (committed) | Side-load (gitignored) |
|---|---|
| `assets/points.bin`, `edges.bin` — CLIP galaxy + webs | `assets/vessel.wswv` (~58 MB) — splat vessel |
| `assets/atlas_0.png` — human-trace photos | `v2/emergence_atlas.png` (~161 MB) — "watch it think" |
| `v2/dream_atlas.png` (31 MB) + `dream_pos.txt` — dreams | `assets/audio/voice_*.wav` (~44 MB) — THEM voices |
| `v2/{stories,credits,captions}_atlas.png` + index/pos | *(`assets/dreams/` raw PNGs — NOT runtime-needed)* |
| `assets/{labels,words,classify,label_words}.txt`, `manifest.json` | |
| `assets/audio/*.wav` — ~192 instrument samples | |

Fallbacks when a side-load asset is absent: vessel → procedural blob; emergence → off (dreams
still pop in resolved from the committed `dream_atlas.png`); voices → synthesized whisper.

## Steps

**1. Clone + submodules, on every node.**
```bash
git clone <allolib_playground remote> && cd allolib_playground
git submodule update --init --recursive
git -C MAT201B_Projects checkout v2 && git -C MAT201B_Projects pull
```
This brings allolib + the committed core assets above.

**2. Bake the side-load assets once (on the Mac), bundle, copy to every node.**
```bash
# on the Mac, after baking (see each script's header):
#   factory/bake_vessel_replicate.py + factory/stage_d_vessel.py  -> assets/vessel.wswv
#   v2/bake_emergence.py                                          -> v2/emergence_atlas.png
#   v2/bake_voices.py   (macOS `say` — Mac-only)                  -> assets/audio/voice_*.wav
bash MAT201B_Projects/reagency/factory/make_dome_bundle.sh        # -> reagency/dome_assets.tgz
# then, on EACH node:
tar xzf dome_assets.tgz -C /path/to/MAT201B_Projects/reagency
```
Voices can only be re-baked on a Mac, so they MUST travel in the bundle to Linux nodes.
*(Skip step 2 entirely if you accept procedural-vessel + emergence-off + synth-voice fidelity.)*

**3. Verify the asset set is byte-identical across nodes** (the renderers draw deterministic,
identical frames only if their inputs match):
```bash
md5sum assets/points.bin assets/edges.bin assets/atlas_0.png assets/vessel.wswv \
       v2/dream_atlas.png v2/emergence_atlas.png   # compare across machines
```

**4. Install build prerequisites on each Linux node:** `cmake >= 3.24`, a C++17 compiler,
and allolib's transitive dev packages (GL / X11 / ALSA). Audio runs only on the primary.

**5. Build once per node.** Preferred (passes an absolute assets path → CWD-independent):
```bash
bash MAT201B_Projects/reagency/run_demo.sh           # builds reagency/CMakeLists.txt, then launches
```
Build-only via the playground builder also works: `./run.sh -n MAT201B_Projects/reagency/src/main_reagency.cpp`.

**6. Author `distributed_app.toml`** (it is gitignored — never commit it):
```bash
cd MAT201B_Projects/reagency
cp distributed_app.toml.example distributed_app.toml   # then edit broadcastAddress + [[node]] hosts
```
Set `broadcastAddress` to the dome subnet broadcast, one `[[node]]` with `rank=0 role="desktop"`
for the primary host and one `rank=1 role="renderer"` per render host (`host=` the real hostnames).
Put an identical copy in the exact cwd each node launches from. **Do not leave a stale/empty toml** —
it forces a primary-less secondary launch → black screen.

**7. Launch the SAME binary on every machine**, each with its cwd = the dir holding that node's
`distributed_app.toml`, passing the absolute assets path as `argv[1]` (mirror `run_demo.sh`):
```bash
"$BIN" /abs/path/to/MAT201B_Projects/reagency/assets
```
Hostname-matched roles make the primary simulate + emit `WoSWState` + render audio; renderers
apply the synced camera pose and draw with omni/dome warp. (Nav input is auto-disabled on
non-primary nodes so the primary's pose is authoritative.)

**8. Confirm sync:** one shared camera/galaxy across all screens, audio only from the primary,
and — with the real ≥3000-point galaxy — the immersive inside-the-cloud camera + THEM credit ring.

## Known hardware trap — `GL_MAX_TEXTURE_SIZE`

`dream_atlas.png` and `emergence_atlas.png` are large single textures. The loaders check the
GPU's `GL_MAX_TEXTURE_SIZE` and **warn-only** if exceeded — allolib then uploads the texture
**silently black** (no fallback). If a dome render node has a smaller max texture than the dev
Mac, those billboards go black even though the file is present. Fix: check the dome GPUs' max,
and if needed re-bake with more columns (smaller per-tile size) — `v2/bake_emergence.py` /
`v2/bake_dreams.py` expose the `COLS` knob for exactly this.

## What's NOT a problem (audited)
- No hardcoded `/Users/` or absolute paths in any shipped runtime source.
- No macOS-only runtime code (the only `say`/Mac dependency is the offline `bake_voices.py`).
- No tracked file exceeds GitHub's 50 MB warning (largest is `dream_atlas.png`, 31 MB).
- DistributedAppWithState is correctly split: primary-only sim + audio; renderers are a pure
  function of the synced `WoSWState`; geometry loads identically per node (never sent over the wire).
