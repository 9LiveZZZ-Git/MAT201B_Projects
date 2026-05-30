#!/usr/bin/env python3
"""
fetch_corpus.py — build a license-clean CC0 museum image corpus for "World of Shadow Work".

Themes: agency / magic / labor (the piece's argument lives in this vocabulary).
Sources (CC0 / public-domain, full attribution captured, no API key):
  - Cleveland Museum of Art  https://openaccess-api.clevelandart.org/  (CC0; DEEP-paginated — the workhorse)
  - Art Institute of Chicago https://api.artic.edu/docs/               (CC0; fast-fail, its IIIF throttles)

Scales to ~10k unique images via deep Cleveland pagination over a broad themed term list.
Images are downsized to <=512px (PIL, if available) so a 10k corpus stays a sane size.

Output (under ./corpus/, images gitignored, ATTRIBUTION.csv tracked):
  corpus/images/<theme>/<source>_<id>.jpg
  corpus/ATTRIBUTION.csv

Re-runnable: existing files/keys are skipped; the run only fills gaps + grows the corpus.

Usage:
  python3 fetch_corpus.py                          # full fetch toward ~10k (all themes, both sources)
  python3 fetch_corpus.py --cma-max 50 --themes magic   # quick test
"""
import argparse
import csv
import io
import os
import sys
import time
import json
import urllib.error
import urllib.parse
import urllib.request

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

UA = {
    "User-Agent": "Mozilla/5.0 (compatible; WorldOfShadowWork/0.1; academic art project; lpfreiburg@ucsb.edu)",
    "Accept": "image/jpeg,image/*,*/*;q=0.8",
    "Referer": "https://www.artic.edu/",
}
ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "corpus")
MAX_EDGE = 512   # downsize longest edge (0 = keep original); set from --max-edge

# Broad, on-theme CC0-museum search terms. Agency leans on concrete proxies (puppets,
# automata, hands, masks) that stage control / will / its loss.
QUERIES = {
    "labor": ["labor", "worker", "factory", "industry", "harvest", "miner", "weaver",
              "blacksmith", "loom", "spinning", "peasant", "forge", "mill", "tools",
              "craftsman", "carpenter", "mason", "potter", "sewing", "textile", "plow",
              "reaper", "servant", "market", "trade", "machine", "engine", "laborer",
              "fishermen", "agriculture", "farmer", "plowing", "threshing", "shepherd",
              "fishing", "hunt", "weaving", "pottery", "glassblowing", "goldsmith",
              "tannery", "brewing", "baking", "spinner", "seamstress", "sailor",
              "builder", "quarry", "mining", "ironwork", "anvil", "kiln", "cobbler",
              "tailor", "butcher", "laundress", "labourers", "workshop", "guild"],
    "magic": ["alchemy", "witchcraft", "tarot", "astrology", "talisman", "amulet",
              "divination", "zodiac", "sorcery", "demon", "ritual", "astrologer",
              "occult", "magic", "spell", "spirit", "ghost", "oracle", "prophecy",
              "incantation", "charm", "idol", "serpent", "moon", "alembic", "philosopher",
              "mystic", "vision", "dream", "transformation", "sorceress", "wizard",
              "magician", "necromancy", "grimoire", "pentagram", "seance", "fortune teller",
              "celestial", "horoscope", "relic", "shrine", "altar", "sacrifice", "exorcism",
              "devil", "angel", "apparition", "enchantress", "conjurer", "occultism",
              "hermetic", "astronomer", "comet", "eclipse"],
    "agency": ["puppet", "marionette", "automaton", "hand", "mask", "gesture", "puppeteer",
               "self-portrait", "fortune", "mechanical", "clockwork", "arm", "grasp", "lever",
               "wheel", "key", "lock", "chain", "crown", "scepter", "throne", "figure", "will",
               "fate", "command", "hands", "puppets", "automata", "mechanical toy", "robot",
               "golem", "effigy", "mannequin", "ventriloquist", "strings", "helm", "tiller",
               "commander", "ruler", "sovereign", "manipulation", "obedience", "doll", "toy"],
}

FIELDS = ["file", "theme", "query", "source", "source_id", "title",
          "creator", "date", "license", "source_url", "image_url"]


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _save(data, path):
    """Write JPEG bytes, downsized to <=MAX_EDGE if PIL is available."""
    if _HAVE_PIL and MAX_EDGE > 0:
        try:
            im = Image.open(io.BytesIO(data)).convert("RGB")
            im.thumbnail((MAX_EDGE, MAX_EDGE))
            im.save(path, "JPEG", quality=88)
            return
        except Exception:
            pass
    with open(path, "wb") as f:
        f.write(data)


def download(url, path, retries=3):
    """Fetch an image with retry+backoff on throttling (AIC IIIF 403s under load)."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 1200:
                raise ValueError("suspiciously small response (%d bytes)" % len(data))
            if data[:2] != b"\xff\xd8":
                raise ValueError("not a JPEG")
            _save(data, path)
            return
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(0.8 + 1.2 * attempt)
                continue
            raise
        except Exception as e:
            last = e
            if attempt + 1 < retries:
                time.sleep(0.4)
    raise last if last else RuntimeError("download failed")


def fetch_aic(theme, q, cap, seen, rows):
    # AIC's image server throttles hard; fast-fail (retries=1) so 403s skip instantly.
    base = "https://api.artic.edu/api/v1/artworks/search"
    params = {"q": q, "query[term][is_public_domain]": "true",
              "fields": "id,title,artist_display,date_display,image_id,is_public_domain",
              "limit": min(cap, 100)}
    try:
        d = get_json(base + "?" + urllib.parse.urlencode(params))
    except Exception as e:
        print("   [AIC] query error %r: %s" % (q, e)); return 0
    n = 0
    for a in d.get("data", []):
        iid = a.get("image_id")
        if not iid or not a.get("is_public_domain"):
            continue
        key = "aic_%s" % a["id"]
        if key in seen:
            continue
        fn = key + ".jpg"
        path = os.path.join(CORPUS, "images", theme, fn)
        img = "https://www.artic.edu/iiif/2/%s/full/600,/0/default.jpg" % iid
        if not os.path.exists(path):
            try:
                download(img, path, retries=1)
            except Exception:
                continue           # throttled -> skip fast, lean on Cleveland
            time.sleep(0.1)
        seen.add(key)
        rows.append({"file": "images/%s/%s" % (theme, fn), "theme": theme, "query": q,
                     "source": "Art Institute of Chicago", "source_id": a["id"],
                     "title": a.get("title", ""),
                     "creator": (a.get("artist_display") or "").replace("\n", ", "),
                     "date": a.get("date_display", ""), "license": "CC0 1.0 (Public Domain)",
                     "source_url": "https://www.artic.edu/artworks/%s" % a["id"], "image_url": img})
        n += 1
    return n


def fetch_cma(theme, q, cap, seen, rows, page=100):
    # Cleveland: deep paginate (skip) up to `cap` NEW images for this term. The workhorse.
    base = "https://openaccess-api.clevelandart.org/api/artworks/"
    n, skip = 0, 0
    while n < cap:
        params = {"q": q, "cc0": 1, "has_image": 1, "limit": page, "skip": skip}
        try:
            d = get_json(base + "?" + urllib.parse.urlencode(params))
        except Exception as e:
            print("   [CMA] query error %r: %s" % (q, e)); break
        data = d.get("data", [])
        if not data:
            break
        for a in data:
            web = ((a.get("images") or {}).get("web") or {}).get("url")
            if not web:
                continue
            key = "cma_%s" % a["id"]
            if key in seen:
                continue
            fn = key + ".jpg"
            path = os.path.join(CORPUS, "images", theme, fn)
            if not os.path.exists(path):
                try:
                    download(web, path)
                except Exception as e:
                    print("   [CMA] dl error %s: %s" % (key, e)); continue
                time.sleep(0.1)
            seen.add(key)
            creators = ", ".join(c.get("description", "") for c in (a.get("creators") or []))
            rows.append({"file": "images/%s/%s" % (theme, fn), "theme": theme, "query": q,
                         "source": "Cleveland Museum of Art", "source_id": a["id"],
                         "title": a.get("title", ""), "creator": creators,
                         "date": a.get("creation_date", ""), "license": "CC0 1.0 (Public Domain)",
                         "source_url": a.get("url", "https://www.clevelandart.org/art/%s" % a.get("accession_number", "")),
                         "image_url": web})
            n += 1
            if n >= cap:
                break
        skip += page
        if len(data) < page:
            break
    return n


def main():
    global MAX_EDGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--cma-max", type=int, default=300, help="max NEW images per term from Cleveland")
    ap.add_argument("--aic-max", type=int, default=40, help="max per term from AIC (throttled)")
    ap.add_argument("--max-edge", type=int, default=512, help="downsize longest edge (0=keep original)")
    ap.add_argument("--themes", nargs="*", default=list(QUERIES.keys()))
    ap.add_argument("--sources", nargs="*", default=["cma", "aic"])
    args = ap.parse_args()
    MAX_EDGE = args.max_edge

    for theme in args.themes:
        os.makedirs(os.path.join(CORPUS, "images", theme), exist_ok=True)

    csv_path = os.path.join(CORPUS, "ATTRIBUTION.csv")
    rows, seen = [], set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                src = r.get("source", "")
                pre = "aic" if "Chicago" in src else ("cma" if "Cleveland" in src else "x")
                seen.add("%s_%s" % (pre, r["source_id"]))

    start = len(seen)
    for theme in args.themes:
        print("== theme: %s ==" % theme)
        for q in QUERIES.get(theme, []):
            got = 0
            if "cma" in args.sources:
                got += fetch_cma(theme, q, args.cma_max, seen, rows)
            if "aic" in args.sources:
                got += fetch_aic(theme, q, args.aic_max, seen, rows)
            print("  %-16s +%d  (corpus now %d)" % (q, got, len(seen)))
        # checkpoint the CSV after each theme (so a long run is crash-safe)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})

    print("\nDONE. %d new this run; %d unique images total." % (len(seen) - start, len(seen)))
    print("Attribution -> %s" % csv_path)


if __name__ == "__main__":
    sys.exit(main())
