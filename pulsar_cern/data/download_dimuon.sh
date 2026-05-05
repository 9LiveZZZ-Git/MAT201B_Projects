#!/bin/bash
# Fetch the full CMS Open Data dimuon CSV (~6 MB, ~100k events).
# Source: CERN Open Data Portal, record 545 (CMS 2010 DoubleMu).
# Columns: Run,Event,Type1,E1,px1,py1,pz1,pt1,eta1,phi1,Q1,Type2,E2,px2,py2,pz2,pt2,eta2,phi2,Q2,M

set -e
DATA_DIR="$(dirname "$0")"
OUT="${DATA_DIR}/dimuon_full.csv"

# Primary URL (Tom McCauley educational mirror, CMS Open Data record 545)
URL="https://opendata.cern.ch/record/545/files/Dimuon_DoubleMu.csv"

# Mirror fallbacks
ALT1="http://opendata.cern.ch/record/700/files/MUO_2010B_Mu.csv"
ALT2="https://github.com/cms-opendata-analyses/DimuonSpectrumNanoAODOutreachAnalysis/raw/master/data/Dimuon_DoubleMu.csv"

echo "Downloading CMS dimuon dataset..."
if curl -fSL "${URL}" -o "${OUT}"; then
    echo "Saved to ${OUT}"
elif curl -fSL "${ALT1}" -o "${OUT}"; then
    echo "Saved (mirror) to ${OUT}"
elif curl -fSL "${ALT2}" -o "${OUT}"; then
    echo "Saved (mirror) to ${OUT}"
else
    echo "All sources failed. Check https://opendata.cern.ch and update URL." >&2
    exit 1
fi

LINES=$(wc -l < "${OUT}")
echo "Loaded ${LINES} lines into ${OUT}"
echo "Point pulsar_cern.cpp at this file (kCsvPath) or rename it to dimuon_sample.csv to override the default."
