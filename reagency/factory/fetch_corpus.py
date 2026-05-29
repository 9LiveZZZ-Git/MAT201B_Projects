#!/usr/bin/env python3
"""
fetch_corpus.py — build a license-clean image corpus for "World of Shadow Work".

Themes: agency / magic / labor (the piece's argument lives in this vocabulary).
Sources (no API key, CC0 / public-domain, full attribution captured):
  - Art Institute of Chicago   https://api.artic.edu/docs/   (data CC0; PD artworks)
  - Cleveland Museum of Art     https://openaccess-api.clevelandart.org/  (CC0)

Output (under ./corpus/, gitignored except ATTRIBUTION.csv):
  corpus/images/<theme>/<source>_<id>.jpg
  corpus/ATTRIBUTION.csv   (file, theme, query, source, source_id, title, creator, date, license, source_url, image_url)

This is OFFLINE factory tooling — NOT part of the allolib build. Stdlib only.
Re-runnable: existing files are skipped; raise --per-query to grow the corpus.

Usage:
  python3 fetch_corpus.py                       # full curated fetch (all themes, both sources)
  python3 fetch_corpus.py --per-query 4 --themes magic   # quick test
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "WorldOfShadowWork/0.1 (MAT201B academic art project; lpfreiburg@ucsb.edu)"}
ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "corpus")

# Search terms per theme. Agency is abstract, so it leans on concrete proxies
# (puppets, automata, hands, masks) that visually stage control / will / its loss.
QUERIES = {
    "labor": ["labor", "worker", "factory", "industry", "harvest", "miner",
              "weaver", "blacksmith", "loom", "spinning", "peasant", "forge",
              "mill", "tools", "craftsman", "carpenter", "mason", "potter",
              "sewing", "textile", "plow", "reaper", "servant", "market",
              "trade", "wage", "machine", "engine", "laborer", "fishermen"],
    "magic": ["alchemy", "witchcraft", "tarot", "astrology", "talisman", "amulet",
              "divination", "zodiac", "sorcery", "demon", "ritual", "astrologer",
              "occult", "magic", "spell", "spirit", "ghost", "oracle", "prophecy",
              "incantation", "charm", "idol", "serpent", "moon", "alembic",
              "philosopher", "mystic", "vision", "dream", "transformation"],
    "agency": ["puppet", "marionette", "automaton", "hand", "mask", "gesture",
               "puppeteer", "self-portrait", "fortune", "mechanical", "clockwork",
               "arm", "grasp", "lever", "wheel", "key", "lock", "chain", "crown",
               "scepter", "throne", "figure", "will", "fate", "command"],
}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def download(url, path):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if len(data) < 1200:          # too small to be a real image (likely an error page)
        raise ValueError("suspiciously small response (%d bytes)" % len(data))
    if data[:2] != b"\xff\xd8":   # JPEG magic
        raise ValueError("not a JPEG")
    with open(path, "wb") as f:
        f.write(data)


def fetch_aic(theme, q, per_query, seen, rows):
    base = "https://api.artic.edu/api/v1/artworks/search"
    params = {
        "q": q,
        "query[term][is_public_domain]": "true",
        "fields": "id,title,artist_display,date_display,image_id,is_public_domain",
        "limit": per_query,
    }
    url = base + "?" + urllib.parse.urlencode(params)
    try:
        d = get_json(url)
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
        img = "https://www.artic.edu/iiif/2/%s/full/400,/0/default.jpg" % iid
        if not os.path.exists(path):
            try:
                download(img, path)
            except Exception as e:
                print("   [AIC] dl error %s: %s" % (key, e)); continue
            time.sleep(0.15)
        seen.add(key)
        rows.append({
            "file": "images/%s/%s" % (theme, fn), "theme": theme, "query": q,
            "source": "Art Institute of Chicago", "source_id": a["id"],
            "title": a.get("title", ""),
            "creator": (a.get("artist_display") or "").replace("\n", ", "),
            "date": a.get("date_display", ""),
            "license": "CC0 1.0 (Public Domain)",
            "source_url": "https://www.artic.edu/artworks/%s" % a["id"],
            "image_url": img,
        })
        n += 1
    return n


def fetch_cma(theme, q, per_query, seen, rows):
    base = "https://openaccess-api.clevelandart.org/api/artworks/"
    params = {"q": q, "cc0": 1, "has_image": 1, "limit": per_query}
    url = base + "?" + urllib.parse.urlencode(params)
    try:
        d = get_json(url)
    except Exception as e:
        print("   [CMA] query error %r: %s" % (q, e)); return 0
    n = 0
    for a in d.get("data", []):
        imgs = a.get("images") or {}
        web = (imgs.get("web") or {}).get("url")
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
            time.sleep(0.15)
        seen.add(key)
        creators = ", ".join(c.get("description", "") for c in (a.get("creators") or []))
        rows.append({
            "file": "images/%s/%s" % (theme, fn), "theme": theme, "query": q,
            "source": "Cleveland Museum of Art", "source_id": a["id"],
            "title": a.get("title", ""), "creator": creators,
            "date": a.get("creation_date", ""),
            "license": "CC0 1.0 (Public Domain)",
            "source_url": a.get("url", "https://www.clevelandart.org/art/%s" % a.get("accession_number", "")),
            "image_url": web,
        })
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=20, help="max images per search term per source")
    ap.add_argument("--themes", nargs="*", default=list(QUERIES.keys()))
    ap.add_argument("--sources", nargs="*", default=["aic", "cma"])
    args = ap.parse_args()

    for theme in args.themes:
        os.makedirs(os.path.join(CORPUS, "images", theme), exist_ok=True)

    csv_path = os.path.join(CORPUS, "ATTRIBUTION.csv")
    rows = []
    # preload already-known keys so re-runs don't duplicate across themes
    seen = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                seen.add("%s_%s" % ("aic" if "Chicago" in r["source"] else "cma", r["source_id"]))

    total = 0
    for theme in args.themes:
        print("== theme: %s ==" % theme)
        for q in QUERIES.get(theme, []):
            got = 0
            if "aic" in args.sources:
                got += fetch_aic(theme, q, args.per_query, seen, rows)
            if "cma" in args.sources:
                got += fetch_cma(theme, q, args.per_query, seen, rows)
            total += got
            print("  %-14s +%d  (corpus now %d)" % (q, got, len(seen)))

    fields = ["file", "theme", "query", "source", "source_id", "title",
              "creator", "date", "license", "source_url", "image_url"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print("\nDONE. fetched %d new this run; %d images total." % (total, len(seen)))
    print("Attribution -> %s" % csv_path)


if __name__ == "__main__":
    sys.exit(main())
