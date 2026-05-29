#!/usr/bin/env python3
"""
prep_vessel_inputs.py — gather the input images for the vessel generator (LGM / TRELLIS)
into work/vessel_inputs/. One .ply gets generated per input image, then stage_d packs them
into vessel.wswv (keyframes morph in that order). More inputs = a richer, longer-cycling
vessel — but more generation time and a bigger bank (stage_d enforces a size cap).

Modes:
  --mode all   (default) EVERY corpus image  -> one vessel keyframe per image (rich; slow)
  --mode reps  one representative image per cluster (work/cluster_reps.json; ~6-24; fast)
  --limit N    cap the number of inputs (0 = no cap)
"""
import argparse
import glob
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "reps"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="cap inputs (0 = no cap)")
    a = ap.parse_args()

    work = os.path.join(HERE, "work")
    corpus = os.path.join(HERE, "corpus")
    out = os.path.join(work, "vessel_inputs")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    srcs = []   # (name, abs_path)
    if a.mode == "reps":
        reps_path = os.path.join(work, "cluster_reps.json")
        if not os.path.exists(reps_path):
            raise SystemExit("[prep] no work/cluster_reps.json — run stage_b_layout.py first")
        for cid, r in json.load(open(reps_path)).items():
            if r.get("file"):
                srcs.append(("cluster_%s" % cid, os.path.join(corpus, r["file"])))
    else:  # all
        for p in sorted(glob.glob(os.path.join(corpus, "images", "**", "*.jpg"), recursive=True)):
            srcs.append((os.path.splitext(os.path.basename(p))[0], p))

    if a.limit > 0:
        srcs = srcs[:a.limit]

    n = 0
    for name, src in srcs:
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out, name + ".jpg"))
            n += 1
    print("[prep] mode=%s -> %d vessel input images in %s" % (a.mode, n, out))
    print("       feed to LGM:  --test_path %s   (=> %d vessel keyframes)" % (out, n))


if __name__ == "__main__":
    main()
