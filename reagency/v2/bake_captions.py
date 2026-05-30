#!/usr/bin/env python3
# bake_captions.py -- World of Shadow Work v2.
# Rasterize the verified me/you/them/us confession lines into a caption atlas the runtime
# CaptionLayer billboards with a typewriter reveal. LOCAL only (PIL) -- no Colab / no GPU.
#
# Output (reagency/v2/):
#   captions_atlas.png   -- white text on transparent; ONE wrapped line per 1024x64 cell, stacked.
#   captions_index.txt   -- one row per line-cell: "captionId person act cellRow charCount"
#                           person: 0=ME 1=YOU 2=THEM 3=US ; act: 1..5 (0 = solo/any)
# The C++ never needs the text (it's in the atlas) -- only this metadata for scheduling + reveal.
import json, os, re, sys
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
# the verified draft from the content workflow (confession_lines: person, line, context)
SRC = ("/private/tmp/claude-504/-Users-matstudents-allolib-playground/"
       "77fb5f6e-8737-49e8-b7ad-14919aab3920/tasks/wciagagnh.output")
lines = json.load(open(SRC))["result"]["draft"]["confession_lines"]

CW, CH = 1024, 64          # cell: 1024 wide x 64 tall (a wide LabelLayer-style cell)
PAD, FS = 18, 30           # left pad, font size
PERSON = {"me": 0, "you": 1, "them": 2, "us": 3}
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}

def load_font():
    for p in ("/System/Library/Fonts/Supplemental/Georgia.ttf",
              "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
              "/Library/Fonts/Georgia.ttf",
              "/System/Library/Fonts/NewYork.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/SFNS.ttf"):
        if os.path.exists(p):
            try: return ImageFont.truetype(p, FS), os.path.basename(p)
            except Exception: pass
    return ImageFont.load_default(), "default"

font, fontname = load_font()
_probe = ImageDraw.Draw(Image.new("RGBA", (CW, CH)))
def width(s): return _probe.textlength(s, font=font)

def wrap(text, maxw):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if width(t) <= maxw or not cur:
            cur = t
        else:
            out.append(cur); cur = w
    if cur: out.append(cur)
    return out

def act_of(person, context):
    m = re.search(r"Act\s+([IVX]+)", context or "")
    if m and m.group(1) in ROMAN: return ROMAN[m.group(1)]
    if "SOLO" in (context or "").upper(): return 0      # fragment / any
    return {"me": 2, "you": 4, "them": 3, "us": 5}.get(person, 0)

# build the wrapped line-cells across all captions
cells = []   # (captionId, person, act, text, charCount, inkRight)
for cid, l in enumerate(lines):
    p = PERSON.get(l["person"], 0)
    a = act_of(l["person"], l.get("context", ""))
    for wl in wrap(l["line"], CW - 2 * PAD):
        ink = min(1.0, (PAD + width(wl)) / CW)   # right edge of the ink, normalized (for the reveal)
        cells.append((cid, p, a, wl, len(wl), ink))

N = len(cells)
atlas = Image.new("RGBA", (CW, CH * N), (0, 0, 0, 0))
draw = ImageDraw.Draw(atlas)
for row, (cid, p, a, text, n, ink) in enumerate(cells):
    y = row * CH + (CH - FS) // 2 - 4
    draw.text((PAD, y), text, font=font, fill=(238, 236, 230, 255))   # warm off-white archival ink

atlas.save(os.path.join(HERE, "captions_atlas.png"))
with open(os.path.join(HERE, "captions_index.txt"), "w") as f:
    f.write("# captionId person act cellRow charCount inkRight  "
            "(person 0=ME 1=YOU 2=THEM 3=US; act 1-5, 0=solo)\n")
    for row, (cid, p, a, text, n, ink) in enumerate(cells):
        f.write("%d %d %d %d %d %.4f\n" % (cid, p, a, row, n, ink))

print("font:", fontname)
print("captions:", len(lines), "| line-cells:", N, "| atlas: %dx%d" % (CW, CH * N))
print("wrote captions_atlas.png + captions_index.txt to", HERE)
