#!/usr/bin/env python3
# bake_emergence.py -- World of Shadow Work v2 (Phase-2, the "watch it think" layer).
# Packs the per-dream EMERGENCE frames (intermediate diffusion latents decoded mid-run = the dream
# forming from noise) into ONE atlas + an index, so the runtime EmergencePlayer can animate a
# dream's formation when the machine ATTENDS its node. LOCAL only (PIL, no GPU).
#
# Output (reagency/v2/):
#   emergence_atlas.png   -- grid of CELL-px frames, all dreams concatenated in node order
#   emergence_index.txt   -- one row per dream: "node startCell numFrames"
import os, json, struct
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
EM = os.path.join(ASSETS, "dreams", "emergence")

CELL, COLS = 160, 46      # 2112 frames -> 7360x7360 (COLS=20 would be 16960 tall > GL_MAX 16384 = silent black)

prov = json.load(open(os.path.join(ASSETS, "dream_provenance.json")))
nodes = sorted(prov.keys(), key=lambda k: int(k))

# gather (node, [frame paths in order]) using the rich emergence_frames list
plan = []
total = 0
for nk in nodes:
    did = prov[nk].get("dream_id", "")
    frames = prov[nk].get("emergence_frames", [])
    paths = []
    for fr in frames:
        p = os.path.join(EM, "%s_%02d.png" % (did, fr))
        if os.path.exists(p):
            paths.append(p)
    plan.append((nk, paths))
    total += len(paths)

ncells = max(1, total)
rows = (ncells + COLS - 1) // COLS
atlas = Image.new("RGBA", (CELL * COLS, CELL * rows), (0, 0, 0, 0))
index = []
cell = 0
for nk, paths in plan:
    start = cell
    for p in paths:
        im = Image.open(p).convert("RGBA").resize((CELL, CELL), Image.LANCZOS)
        c, r = cell % COLS, cell // COLS
        atlas.paste(im, (c * CELL, r * CELL))
        cell += 1
    index.append((int(nk), start, len(paths)))

atlas.save(os.path.join(HERE, "emergence_atlas.png"))
with open(os.path.join(HERE, "emergence_index.txt"), "w") as f:
    f.write("# node startCell numFrames  (CELL=%d COLS=%d; frames in formation order)\n" % (CELL, COLS))
    for node, start, n in index:
        f.write("%d %d %d\n" % (node, start, n))

print("dreams:", len(index), "| total frames:", total, "| atlas: %dx%d cell=%d cols=%d" %
      (atlas.size[0], atlas.size[1], CELL, COLS))
print("wrote emergence_atlas.png + emergence_index.txt to", HERE)
