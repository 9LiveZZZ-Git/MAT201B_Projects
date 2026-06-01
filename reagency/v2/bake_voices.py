#!/usr/bin/env python3
# bake_voices.py -- World of Shadow Work, Phase-2 2a: the REAL ghostly voices (THEM made literal).
#
# TTS the 142 worker one-liners (v2/stories.json) into assets/audio/voice_NNN.wav via macOS `say`
# (LOCAL, offline, no GPU). The runtime's loadSamples router picks these up as the rVoice_ role
# (filename contains "voice"; AudioEngine.cpp: has("voice") && !has("voicing") -> non-pitched), so on
# arrival the EV_WHISPER layer-4 path plays a real speech sample that shares the duck + cap_ capture +
# granular shred -> the machine *editing the human record*. This is a working BASELINE; swap in
# higher-quality TTS / LibriVox / CC0 murmur later by replacing assets/audio/voice_*.wav.
#
# These WAVs are regenerable (script + stories.json) -> gitignored, not committed. Re-run anytime.
import json, os, re, subprocess, sys

HERE   = os.path.dirname(os.path.abspath(__file__))         # reagency/v2
ROOT   = os.path.dirname(HERE)                              # reagency
STORIES = os.path.join(HERE, "stories.json")
OUT     = os.path.join(ROOT, "assets", "audio")

VOICE = os.environ.get("WSW_VOICE", "")     # e.g. WSW_VOICE=Daniel ; "" = system default
RATE  = int(os.environ.get("WSW_RATE", "172"))   # measured testimony delivery
MAXC  = 140                                  # cap each line so the sample bank stays light/punchy

def clean(t):
    t = re.sub(r"https?://\S+", "", t or "").replace("—", " - ").strip()
    if len(t) > MAXC:                                       # trim to a word boundary
        t = t[:MAXC].rsplit(" ", 1)[0] + "..."
    return t

stories = json.load(open(STORIES))
os.makedirs(OUT, exist_ok=True)
# clear any prior bake so a shorter stories.json doesn't leave orphans
for f in os.listdir(OUT):
    if f.startswith("voice_") and f.endswith(".wav"):
        os.remove(os.path.join(OUT, f))

n = 0
for i, st in enumerate(stories):
    line = clean(st.get("oneLiner", "") or st.get("figure", ""))
    if not line:
        continue
    dst = os.path.join(OUT, "voice_%03d.wav" % i)
    cmd = ["say", "-o", dst, "--data-format=LEI16@22050", "--file-format=WAVE"]
    if VOICE:
        cmd += ["-v", VOICE]
    cmd += ["-r", str(RATE), line]
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        n += 1
    except Exception as e:
        print("  ERR voice_%03d: %s" % (i, e), flush=True)
    if (i + 1) % 25 == 0:
        print("  %d/%d ..." % (i + 1, len(stories)), flush=True)

total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.startswith("voice_"))
print("baked %d voice WAVs -> %s  (%.1f MB)  voice=%s rate=%d" % (n, OUT, total / 1e6, VOICE or "default", RATE))
print("the runtime loads these as rVoice_ (Phase-2 2a). NOTE: voice_*.wav are gitignored (regenerable).")
