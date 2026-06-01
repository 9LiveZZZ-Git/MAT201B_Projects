#!/usr/bin/env python3
# bake_vessel_replicate.py -- generate dense TRELLIS Gaussian-splat keyframes via Replicate (HOSTED GPU),
# entirely from this Mac (NO local CUDA). Per corpus image -> firtoz/trellis -> a gaussian .ply (or a GLB
# we sample into a dense colored point cloud) -> reagency/factory/work/vessels/. Then pack with
# stage_d_vessel.py -> reagency/assets/vessel.wswv (DENS=4 in VesselSplats.cpp pairs with G=30000).
#
# SETUP (on the Mac):
#   pip install replicate trimesh numpy
#   export REPLICATE_API_TOKEN=...        # from https://replicate.com/account/api-tokens
# TEST CHEAP FIRST (4 images, ~$0.4), confirm .ply land, THEN scale:
#   WSW_N=4 python3 MAT201B_Projects/reagency/factory/bake_vessel_replicate.py
#   python3      MAT201B_Projects/reagency/factory/bake_vessel_replicate.py     # full 128
# Cost: firtoz/trellis is ~$0.05-0.10/run on an A100 -> ~$6-13 for 128. Re-runnable (skips done plys).
import os, glob, sys, urllib.request

HERE   = os.path.dirname(os.path.abspath(__file__))         # reagency/factory
CORPUS = os.path.join(HERE, "corpus", "images")
OUT    = os.path.join(HERE, "work", "vessels"); os.makedirs(OUT, exist_ok=True)

N_IMAGES = int(os.environ.get("WSW_N", "128"))
STEPS    = int(os.environ.get("WSW_STEPS", "25"))           # ss + slat sampling steps (quality; Replicate caps ~50)
GPTS     = int(os.environ.get("WSW_GLB_PTS", "40000"))      # points sampled if a build returns only a GLB
MODEL    = "firtoz/trellis"

if not os.environ.get("REPLICATE_API_TOKEN"):
    sys.exit("set REPLICATE_API_TOKEN first (https://replicate.com/account/api-tokens)")
try:
    import replicate
except ImportError:
    sys.exit("pip install replicate trimesh numpy")

imgs = sorted(glob.glob(os.path.join(CORPUS, "**", "*.jpg"), recursive=True))
if not imgs:
    sys.exit(f"no corpus images in {CORPUS} (run factory/download_from_csv.py, or point CORPUS at your images)")
picks = imgs[:: max(1, len(imgs) // N_IMAGES)][:N_IMAGES]   # evenly across the corpus (stage_a node order)
print(f"{len(picks)} images -> {MODEL} (steps={STEPS})  out={OUT}")


def collect_urls(o, acc):
    if o is None: return
    if hasattr(o, "url"): acc.append(o.url)
    elif isinstance(o, str): acc.append(o)
    elif isinstance(o, dict): [collect_urls(v, acc) for v in o.values()]
    elif isinstance(o, (list, tuple)): [collect_urls(v, acc) for v in o]


def glb_to_ply(glb_path, dst, n):
    """GLB textured mesh -> dense colored point cloud .ply (stage_d reads x,y,z,red,green,blue)."""
    import trimesh, numpy as np
    m = trimesh.load(glb_path, force="mesh")
    pts, fid = trimesh.sample.sample_surface(m, n)
    rgb = None
    try:
        vc = m.visual.to_color().vertex_colors                 # (V,4)
        if vc is not None and len(vc) == len(m.vertices):
            rgb = vc[m.faces[fid]][:, :, :3].mean(axis=1).astype("uint8")   # mean of each face's 3 vertex colors
    except Exception:
        pass
    if rgb is None:
        rgb = np.full((len(pts), 3), 180, "uint8")
    a = np.full((len(pts), 1), 255, "uint8")
    trimesh.PointCloud(pts, colors=np.hstack([rgb, a])).export(dst)


base = dict(ss_sampling_steps=STEPS, slat_sampling_steps=STEPS,
            generate_model=True, generate_color=False, generate_normal=False, seed=1)
try_gauss = True                                            # try save_gaussian_ply; drop it if the model rejects it

for i, ip in enumerate(picks):
    name = os.path.splitext(os.path.basename(ip))[0]
    dst  = os.path.join(OUT, f"{name}.ply")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        continue
    inp = dict(base)
    if try_gauss: inp["save_gaussian_ply"] = True
    try:
        with open(ip, "rb") as f:
            inp["image"] = f
            out = replicate.run(MODEL, input=inp)
    except Exception as e:
        if try_gauss and "save_gaussian_ply" in str(e):       # unknown input -> drop for all, retry this one
            try_gauss = False; inp.pop("save_gaussian_ply", None)
            try:
                with open(ip, "rb") as f:
                    inp["image"] = f; out = replicate.run(MODEL, input=inp)
            except Exception as e2:
                print(f"[{i+1}/{len(picks)}] ERR {name}: {e2}"); continue
        else:
            print(f"[{i+1}/{len(picks)}] ERR {name}: {e}"); continue

    urls = []; collect_urls(out, urls)
    ply = next((u for u in urls if str(u).lower().endswith(".ply")), None)
    glb = next((u for u in urls if str(u).lower().endswith(".glb")), None)
    if ply:
        urllib.request.urlretrieve(ply, dst)
        print(f"[{i+1}/{len(picks)}] {name}.ply (gaussian) {os.path.getsize(dst)/1e6:.1f} MB")
    elif glb:
        gtmp = os.path.join(OUT, f"{name}.glb"); urllib.request.urlretrieve(glb, gtmp)
        try:
            glb_to_ply(gtmp, dst, GPTS); os.remove(gtmp)
            print(f"[{i+1}/{len(picks)}] {name}.ply (GLB->{GPTS} pts) {os.path.getsize(dst)/1e6:.1f} MB")
        except Exception as e:
            print(f"[{i+1}/{len(picks)}] GLB-convert ERR {name}: {e}  (outputs: {urls})")
    else:
        print(f"[{i+1}/{len(picks)}] no .ply/.glb in output for {name}: {urls}")

n = len(glob.glob(os.path.join(OUT, "*.ply")))
print(f"\n{n} plys in {OUT}")
print("pack -> assets/vessel.wswv with:")
print(f"  python3 {os.path.join(HERE, 'stage_d_vessel.py')} --plys work/vessels --G 30000 --max-mb 80")
