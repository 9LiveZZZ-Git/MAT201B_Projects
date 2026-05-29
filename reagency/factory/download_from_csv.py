#!/usr/bin/env python3
"""
download_from_csv.py — fetch the corpus images directly from ATTRIBUTION.csv.

fetch_corpus.py is re-entrant on (CSV + files) together, but in a fresh clone the
images are gitignored while ATTRIBUTION.csv ships — so its `seen` set (preloaded
from the CSV) skips every download. This downloads straight from each row's
`image_url`, reproducing the exact curated 704-image corpus with no API re-query
drift. Stdlib only; re-runnable (existing files skipped).
"""
import csv
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
UA = {"User-Agent": "WorldOfShadowWork/0.1 (MAT201B academic art project; lpfreiburg@ucsb.edu)"}


def main():
    csv_path = os.path.join(CORPUS, "ATTRIBUTION.csv")
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    n_ok = n_skip = n_err = 0
    for i, r in enumerate(rows):
        rel = r["file"]
        url = r.get("image_url", "").strip()
        path = os.path.join(CORPUS, rel)
        if not url:
            n_err += 1
            continue
        if os.path.exists(path) and os.path.getsize(path) > 0:
            n_skip += 1
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(path, "wb") as f:
                f.write(data)
            n_ok += 1
            time.sleep(0.1)
        except Exception as e:
            n_err += 1
            print("  ERR %s: %s" % (rel, e), flush=True)
        if (i + 1) % 50 == 0:
            print("  %d/%d (ok=%d skip=%d err=%d)" % (i + 1, len(rows), n_ok, n_skip, n_err), flush=True)
    print("DONE. downloaded=%d skipped=%d errors=%d total=%d" % (n_ok, n_skip, n_err, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
