#!/usr/bin/env python3
# bake_stories.py -- World of Shadow Work v2.
# Rasterize the verified labor STORIES into a galaxy atlas + positions: each story is a short
# legible text block placed AT a point in the galaxy (sampled from the real cloud), billboarded by
# the runtime StoryLayer and read by proximity as you fly past. LOCAL only (PIL) -- no Colab.
#
# Input  (first that exists):  stories.json  (list of {"oneLiner": str, "era": str})   <- research output
#                              else them.txt rows (placeholder until the deep-research corpus lands)
# Output (reagency/v2/):  stories_atlas.png   (warm-white text on transparent, stacked line-cells)
#                         stories_pos.txt      "storyId era x y z startRow numLines"
import json, os, struct, sys, random
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")
CW, CH, PAD, FS = 980, 56, 16, 26     # cell w/h, pad, font size
SEED = 42

def load_font():
    for p in ("/System/Library/Fonts/Supplemental/Georgia.ttf",
              "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
              "/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"):
        if os.path.exists(p):
            try: return ImageFont.truetype(p, FS), os.path.basename(p)
            except Exception: pass
    return ImageFont.load_default(), "default"
font, fontname = load_font()
_d = ImageDraw.Draw(Image.new("RGBA", (CW, CH)))
def width(s): return _d.textlength(s, font=font)

TWEET = 280                                  # "twitter rule": every quote/story fits in a tweet
def tweet(s):
    s = " ".join(s.split())
    if len(s) <= TWEET: return s
    cut = s[:TWEET]
    sp = cut.rfind(" ")                      # break on a word boundary, not mid-word
    return (cut[:sp] if sp > 40 else cut).rstrip(" ,;:-") + "…"

def wrap(text, maxw):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if width(t) <= maxw or not cur: cur = t
        else: out.append(cur); cur = w
    if cur: out.append(cur)
    return out

# ---- gather the story one-liners ----
stories = []
sj = os.path.join(HERE, "stories.json")
if os.path.exists(sj):
    for r in json.load(open(sj)):
        ol = (r.get("oneLiner") or "").strip()
        ol = tweet(ol)                                   # "twitter rule": cap to a tweet so the whole passage fits
        if ol: stories.append((ol, r.get("era", "contemporary")))
    src = "stories.json (%d)" % len(stories)
else:  # placeholder: parse them.txt data rows into one-liners
    for line in open(os.path.join(HERE, "them.txt")):
        line = line.rstrip()
        if not line or line.startswith("#"): continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            role, place, rest = parts[0], parts[1], " ; ".join(parts[2:])
            stories.append(("%s, %s -- %s" % (place, role, rest), "contemporary"))
    src = "them.txt placeholder (%d)" % len(stories)
assert stories, "no stories found (add stories.json or v2/them.txt)"

# ---- positions: sample well-spread points from the real galaxy cloud (points.bin / WSWP) ----
def galaxy_positions(n):
    pb = os.path.join(ASSETS, "points.bin")
    if not os.path.exists(pb):
        random.seed(SEED)
        return [(random.uniform(-5,5), random.uniform(-3,3), random.uniform(-5,5)) for _ in range(n)]
    import numpy as np
    raw = open(pb, "rb").read()
    ver, N, stride = struct.unpack_from("<iii", raw, 4)        # after 4-byte 'WSWP'
    a = np.frombuffer(raw, dtype="<f4", offset=16, count=N*stride).reshape(N, stride)
    pos = a[:, :3]
    rad = np.sqrt((pos * pos).sum(1))                          # keep stories in the band the camera flies through
    band = [i for i in range(N) if 1.3 <= rad[i] <= 4.5]       # (was full radius 0.66..6.88 -> many unreachable/illegible)
    if len(band) < n: band = list(range(N))
    random.seed(SEED)
    idx = random.sample(band, min(len(band), n*40))            # candidate pool
    # greedy farthest-point spread so stories don't pile up
    chosen = [idx[0]]
    for _ in range(min(n, len(idx)) - 1):
        best, bestd = -1, -1.0
        for c in idx:
            if c in chosen: continue
            d = min(float(((pos[c]-pos[ch])**2).sum()) for ch in chosen)
            if d > bestd: bestd, best = d, c
        chosen.append(best)
    return [tuple(float(v) for v in pos[c]) for c in chosen[:n]]

positions = galaxy_positions(len(stories))

# ---- build the atlas (one wrapped line per cell, stacked) + positions file ----
cells = []   # (storyId, era, text)
for sid, (ol, era) in enumerate(stories):
    for wl in wrap(ol, CW - 2*PAD):
        cells.append((sid, era, wl))
import math
N = len(cells)
COLS = min(8, max(1, math.ceil(N / 140)))     # grid-pack so the atlas stays under the GPU max (~8192)
ROWS = math.ceil(N / COLS)
atlas = Image.new("RGBA", (CW*COLS, CH*ROWS), (0,0,0,0))
draw = ImageDraw.Draw(atlas)
for idx, (sid, era, text) in enumerate(cells):     # idx is the FLAT cell index; runtime maps -> (col,row)
    col, row = idx % COLS, idx // COLS
    draw.text((col*CW + PAD, row*CH + (CH-FS)//2 - 2), text, font=font, fill=(236, 232, 224, 255))
atlas.save(os.path.join(HERE, "stories_atlas.png"))

with open(os.path.join(HERE, "stories_pos.txt"), "w") as f:
    f.write("# storyId era(0=contemp,1=hist) x y z startRow numLines\n")
    row = 0
    for sid, (ol, era) in enumerate(stories):
        k = sum(1 for c in cells if c[0] == sid)
        x, y, z = positions[sid]
        f.write("%d %d %.4f %.4f %.4f %d %d\n" % (sid, 1 if era == "historical" else 0, x, y, z, row, k))
        row += k

print("source:", src, "| font:", fontname)
print("stories:", len(stories), "| line-cells:", N, "| grid: %dcols x %drows | atlas: %dx%d"
      % (COLS, ROWS, CW*COLS, CH*ROWS))
print("wrote stories_atlas.png + stories_pos.txt to", HERE)
