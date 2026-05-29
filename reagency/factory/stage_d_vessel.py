#!/usr/bin/env python3
"""
stage_d_vessel.py — pack image-to-3D Gaussian-splat .ply keyframes into one compact
WSWV vessel bank the allolib runtime morphs live (NO runtime ML).

Input:  a dir of .ply files (one per cluster/concept), e.g. factory/work/vessels/*.ply
        from LGM (3DTopia/LGM) or DreamGaussian/TriplaneGaussian.
Output: ../assets/vessel.wswv

Imposes a CANONICAL point ordering across all keyframes (Morton / Z-order over a shared
normalized AABB) so the runtime's fixed-index lerp is a real morph, not swimming. Enforces
an identical G across keyframes. See VESSEL.md.

Deps: numpy, plyfile  (pip install numpy plyfile)
"""
import argparse
import glob
import os
import struct

import numpy as np

C0 = 0.28209479177387814  # SH band-0 constant (DC color -> rgb)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def read_ply_gaussians(path):
    """Return (pos[N,3], rgb[N,3] in 0..1, opacity[N], sigma[N]) with activations undone.
    Handles LGM (14-field, DC-only) and Inria/graphdeco (.ply with f_rest_*) schemas."""
    from plyfile import PlyData
    v = PlyData.read(path)['vertex']
    names = set(v.data.dtype.names)
    pos = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)

    if {'f_dc_0', 'f_dc_1', 'f_dc_2'} <= names:          # SH DC color -> rgb (ignore f_rest_*)
        fdc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1).astype(np.float64)
        rgb = np.clip(0.5 + C0 * fdc, 0.0, 1.0)
    elif {'red', 'green', 'blue'} <= names:
        rgb = np.stack([v['red'], v['green'], v['blue']], 1).astype(np.float64) / 255.0
    else:
        rgb = np.full((len(pos), 3), 0.7)

    op = _sigmoid(np.asarray(v['opacity'], np.float64)) if 'opacity' in names else np.ones(len(pos))

    if {'scale_0', 'scale_1', 'scale_2'} <= names:        # log-scale -> exp; isotropic sigma
        sc = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1).astype(np.float64))
        sigma = sc.mean(1)
    else:
        sigma = np.full(len(pos), 0.02)
    return pos, rgb, op, sigma


def pick_fixed_G(pos, rgb, op, sigma, G, rng):
    """Select exactly G Gaussians: top-G by area-aware significance (opacity x sigma);
    pad by resampling the most-significant ones if a keyframe has fewer than G."""
    sig = op * sigma
    n = len(pos)
    if n >= G:
        idx = np.argpartition(-sig, G - 1)[:G]
    else:
        top = np.argsort(-sig)[:max(1, n // 4)]
        pad = rng.choice(top, G - n)
        idx = np.concatenate([np.arange(n), pad])
    return pos[idx], rgb[idx], op[idx], sigma[idx]


def morton_order(pos01):
    """Canonical Z-order over positions in [0,1]^3 (10 bits/axis -> 30-bit code)."""
    q = np.clip((pos01 * 1023.0).astype(np.uint64), 0, 1023)

    def split3(a):
        a = a & np.uint64(0x3ff)
        a = (a | (a << np.uint64(16))) & np.uint64(0x30000ff)
        a = (a | (a << np.uint64(8)))  & np.uint64(0x300f00f)
        a = (a | (a << np.uint64(4)))  & np.uint64(0x30c30c3)
        a = (a | (a << np.uint64(2)))  & np.uint64(0x9249249)
        return a

    code = split3(q[:, 0]) | (split3(q[:, 1]) << np.uint64(1)) | (split3(q[:, 2]) << np.uint64(2))
    return np.argsort(code, kind='stable')


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--plys", default=os.path.join(here, "work", "vessels"))
    ap.add_argument("--assets", default=os.path.abspath(os.path.join(here, "..", "assets")))
    ap.add_argument("--G", type=int, default=12000, help="Gaussians per keyframe (dome-safe ~8k-16k)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.assets, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(a.plys, "*.ply")))
    if not paths:
        raise SystemExit("[stage_d] no .ply files in %s — generate vessels first (see VESSEL.md)" % a.plys)
    print("[stage_d] %d keyframe .ply files" % len(paths))

    kfs = []
    for p in paths:
        pos, rgb, op, sigma = read_ply_gaussians(p)
        pos = pos - pos.mean(0)                       # recenter each
        kfs.append(pick_fixed_G(pos, rgb, op, sigma, a.G, rng))
        print("   %-40s %d -> %d gaussians" % (os.path.basename(p), len(op), a.G))

    K = len(kfs)
    G = a.G
    # ONE global AABB across all keyframes (so the morph never pops in scale)
    allpos = np.concatenate([k[0] for k in kfs], 0)
    mn, mx = allpos.min(0), allpos.max(0)
    center = (mn + mx) / 2.0
    half = float(np.max(mx - center))
    half = max(half, 1e-6)

    bank = np.zeros((K, G, 8), np.float32)
    for k, (pos, rgb, op, sigma) in enumerate(kfs):
        pn = (pos - center) / half                    # ~[-1,1], shared frame
        order = morton_order((pn + 1.0) * 0.5)         # canonical ordering for index-lerp
        bank[k, :, 0:3] = pn[order]
        bank[k, :, 3:6] = rgb[order]
        bank[k, :, 6]   = op[order]
        bank[k, :, 7]   = (sigma[order] / half)        # sigma in normalized units

    aabb = np.concatenate([mn, mx]).astype('<f4')
    out = os.path.join(a.assets, "vessel.wswv")
    with open(out, "wb") as f:
        f.write(b"WSWV")
        f.write(struct.pack("<iiii", 1, K, G, 8))
        f.write(aabb.tobytes())
        f.write(bank.astype(np.float16).tobytes())
    print("[stage_d] DONE -> %s  (K=%d, G=%d, %.2f MB)"
          % (out, K, G, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
