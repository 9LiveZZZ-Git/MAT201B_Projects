#!/usr/bin/env python3
# Generate the LabelLayer assets LOCALLY with NO ML — just text rendering (PIL) + procedural
# placement — so the word visuals (floating words IN the galaxy + the word BY each surfacing photo)
# light up now, before the CLIP regen. The Colab regen's stage_b overwrites these with the real
# semantic classification later. Reads assets/points.bin + assets/words.txt.
import os, struct, sys
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets"))
COLS, CW, CH = 8, 256, 64
N_GALAXY = 36   # how many words to float in the galaxy (subtle)


def _font(size):
    for p in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except Exception: pass
    return ImageFont.load_default()


def read_points(path):
    with open(path, "rb") as f:
        magic = f.read(4); ver, n, stride = struct.unpack("<iii", f.read(12))
        if magic != b"WSWP" or stride != 10:
            raise SystemExit("not a WSWP points.bin")
        return [struct.unpack("<10f", f.read(40)) for _ in range(n)]


def main():
    pts = read_points(os.path.join(ASSETS, "points.bin"))
    words = [w.strip() for w in open(os.path.join(ASSETS, "words.txt")) if w.strip()]
    if not words:
        raise SystemExit("no words.txt")
    hero = [i for i, r in enumerate(pts) if int(round(r[8])) == 0 and int(round(r[9])) >= 0]  # type=image, has atlas
    # spread some nodes across the cloud for floating galaxy labels
    step = max(1, len(pts) // N_GALAXY)
    galaxy = list(range(0, len(pts), step))[:N_GALAXY]

    slot = {}                                   # word -> slot
    def slot_of(w): return slot.setdefault(w, len(slot))
    node_word = {}
    for k, i in enumerate(hero):    node_word[i] = words[(k * 1) % len(words)]
    gal = [(i, words[(j * 7 + 3) % len(words)]) for j, i in enumerate(galaxy)]
    for w in node_word.values(): slot_of(w)
    for _, w in gal:             slot_of(w)

    rows = max(1, (len(slot) + COLS - 1) // COLS)
    atlas = Image.new("RGBA", (COLS * CW, rows * CH), (0, 0, 0, 0)); d = ImageDraw.Draw(atlas)
    for w, s in slot.items():
        gx, gy = (s % COLS) * CW, (s // COLS) * CH
        f = _font(40)
        while getattr(f, "getlength", lambda t: 0)(w) > CW * 0.92 and f.size > 12: f = _font(f.size - 4)
        d.text((gx + CW / 2, gy + CH / 2), w, fill=(255, 255, 255, 255), font=f, anchor="mm")
    atlas.save(os.path.join(ASSETS, "labels_atlas.png"))

    with open(os.path.join(ASSETS, "labels.txt"), "w") as f:        # floating galaxy words: cx cy cz slot word
        for i, w in gal:
            r = pts[i]; f.write("%.3f %.3f %.3f %d %s\n" % (r[0], r[1], r[2], slot[w], w))
    with open(os.path.join(ASSETS, "classify.txt"), "w") as f:      # node -> slot (word by each surfacing photo)
        for i, w in node_word.items(): f.write("%d %d\n" % (i, slot[w]))
    with open(os.path.join(ASSETS, "label_words.txt"), "w") as f:   # slot -> word (whisper reads this)
        for w, s in sorted(slot.items(), key=lambda kv: kv[1]): f.write("%d %s\n" % (s, w))
    print("[gen_labels] %d slots, %d galaxy labels, %d hero classifications -> assets/" % (len(slot), len(gal), len(node_word)))


if __name__ == "__main__":
    main()
