#!/usr/bin/env python3
# bake_vessel_trellis_mac.py -- generate the WoSW vessel keyframes LOCALLY on this Apple Silicon Mac
# via trellis-mac (https://github.com/shivampkumar/trellis-mac), no cloud GPU / no API cost.
#
# trellis-mac is a CUDA-free MPS port of Microsoft TRELLIS.2. IMPORTANT: it outputs a textured GLB
# MESH per image (NOT gaussian splats). The vessel only needs dense colored 3D points, so we sample
# each mesh surface into a colored point-cloud .ply -> stage_d_vessel.py packs vessel.wswv. Visually
# the vessel is the same dense colored morphing cloud; the "splats" are mesh-surface-sampled points.
#
# PREREQS (one-time -- YOU do these; I can't drive a 15 GB model download + Metal build + HF auth):
#   - HuggingFace gated access (instant approval): facebook/dinov3-vitl16-pretrain-lvd1689m, briaai/RMBG-2.0
#   - cd trellis-mac && bash setup.sh   (Python 3.11+, ~15 GB weights, Xcode Metal toolchain, 24 GB+ RAM)
#   - confirm it runs: <trellis-mac>/.venv/bin/python generate.py some.png
#   - in the python running THIS driver:  pip install trimesh numpy
#
# RUN (point at your trellis-mac clone; ~5 min/image -> ~11 h for 128; resumable -- skips done .ply):
#   WSW_TRELLIS_DIR=/path/to/trellis-mac python3 MAT201B_Projects/reagency/factory/bake_vessel_trellis_mac.py
#   # cheap smoke test first (2 images): WSW_N=2 WSW_TRELLIS_DIR=... python3 ...
# THEN pack to spec (128 keyframes x G=30000):
#   python3 MAT201B_Projects/reagency/factory/stage_d_vessel.py --plys work/vessels --G 30000 --max-mb 80
import os, glob, sys, subprocess

HERE   = os.path.dirname(os.path.abspath(__file__))   # reagency/factory
CORPUS = os.path.join(HERE, "corpus", "images")
OUT    = os.path.join(HERE, "work", "vessels"); os.makedirs(OUT, exist_ok=True)

N_IMAGES = int(os.environ.get("WSW_N", "128"))
PIPELINE = os.environ.get("WSW_PIPELINE", "1024_cascade")   # 512 | 1024 | 1024_cascade (highest quality)
TEXSIZE  = os.environ.get("WSW_TEXSIZE", "2048")            # 512 | 1024 | 2048
GPTS     = int(os.environ.get("WSW_GLB_PTS", "40000"))      # surface points sampled per mesh (keep >= stage_d --G)
TRELLIS  = os.environ.get("WSW_TRELLIS_DIR", "")

if not TRELLIS or not os.path.isdir(TRELLIS):
    sys.exit("set WSW_TRELLIS_DIR=/path/to/trellis-mac  (clone it + `bash setup.sh` per its README first)")
PY = os.path.join(TRELLIS, ".venv", "bin", "python")
if not os.path.exists(PY):
    PY = os.environ.get("WSW_TRELLIS_PY", "python3")        # fallback: WSW_TRELLIS_PY=/abs/path/to/venv/python
GEN = os.path.join(TRELLIS, "generate.py")
if not os.path.exists(GEN):
    sys.exit(f"generate.py not found in {TRELLIS} (is WSW_TRELLIS_DIR the trellis-mac clone?)")
try:
    import trimesh, numpy as np  # noqa: F401
except ImportError:
    sys.exit("pip install trimesh numpy   (in the python running THIS driver, not trellis-mac's venv)")

imgs = sorted(glob.glob(os.path.join(CORPUS, "**", "*.jpg"), recursive=True))
if not imgs:
    sys.exit(f"no corpus images in {CORPUS}")
picks = imgs[:: max(1, len(imgs) // N_IMAGES)][:N_IMAGES]   # evenly across agency/labor/magic (stage_a node order)
print(f"{len(picks)} images -> trellis-mac (pipeline={PIPELINE}, tex={TEXSIZE})  out={OUT}")


def glb_to_ply(glb_path, dst, n):
    """GLB textured mesh -> dense colored point cloud .ply (stage_d reads x,y,z,red,green,blue)."""
    import trimesh, numpy as np
    m = trimesh.load(glb_path, force="mesh")
    pts, fid = trimesh.sample.sample_surface(m, n)
    rgb = None
    try:
        vc = m.visual.to_color().vertex_colors               # (V,4)
        if vc is not None and len(vc) == len(m.vertices):
            rgb = vc[m.faces[fid]][:, :, :3].mean(axis=1).astype("uint8")   # mean of each face's 3 vertex colors
    except Exception:
        pass
    if rgb is None:
        rgb = np.full((len(pts), 3), 180, "uint8")
    a = np.full((len(pts), 1), 255, "uint8")
    trimesh.PointCloud(pts, colors=np.hstack([rgb, a])).export(dst)


for i, ip in enumerate(picks):
    name = os.path.splitext(os.path.basename(ip))[0]
    dst  = os.path.join(OUT, f"{name}.ply")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        continue                                             # resumable
    stem = os.path.join(OUT, name)                           # generate.py appends .glb
    glb  = stem + ".glb"
    cmd = [PY, GEN, os.path.abspath(ip), "--output", stem,
           "--pipeline-type", PIPELINE, "--texture-size", TEXSIZE, "--seed", "42"]
    print(f"[{i+1}/{len(picks)}] {name}: generating (~5 min)...", flush=True)
    try:
        subprocess.run(cmd, cwd=TRELLIS, check=True)
    except Exception as e:
        print(f"[{i+1}/{len(picks)}] generate ERR {name}: {e}"); continue
    if not os.path.exists(glb):
        print(f"[{i+1}/{len(picks)}] no GLB produced for {name} (expected {glb})"); continue
    try:
        glb_to_ply(glb, dst, GPTS); os.remove(glb)
        print(f"[{i+1}/{len(picks)}] -> {name}.ply ({GPTS} pts, {os.path.getsize(dst)/1e6:.1f} MB)")
    except Exception as e:
        print(f"[{i+1}/{len(picks)}] GLB->ply ERR {name}: {e}")

n = len(glob.glob(os.path.join(OUT, "*.ply")))
print(f"\n{n} plys in {OUT}")
print("pack -> assets/vessel.wswv:")
print(f"  python3 {os.path.join(HERE, 'stage_d_vessel.py')} --plys work/vessels --G 30000 --max-mb 80")
