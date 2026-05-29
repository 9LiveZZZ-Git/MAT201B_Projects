#!/usr/bin/env python3
"""
generate_words.py — build a LARGE, multilingual, on-theme concept vocabulary for the galaxy.

Expands on-theme seed concepts (agency / magic / labor + Illich) via WordNet synsets (+ one
level of hyponyms for breadth), then Open Multilingual WordNet (~30+ languages) lemmas, into
corpus/words/concepts.txt. Target: well over 2x the image count, all on-theme.

IMPORTANT — multilingual + CLIP: an English-only CLIP (ViT-L/14) embeds non-English text by
SCRIPT, not meaning, so the words wouldn't classify images correctly. For the multilingual
vision, embed with a MULTILINGUAL CLIP in stage_a, e.g.:
    python3 stage_a_embed.py --model xlm-roberta-base-ViT-B-32 --pretrained laion5b_s13b_b90k
With an English-only CLIP, run this with --langs eng.

Needs: nltk (pip install nltk). Downloads 'wordnet' + 'omw-1.4' on first run.
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# On-theme English seed terms (single words resolve best in WordNet).
SEEDS = {
    "agency": ["agency", "will", "intention", "choice", "control", "autonomy", "freedom",
               "power", "hand", "puppet", "marionette", "automaton", "master", "command",
               "obey", "machine", "tool", "gesture", "mask", "fate", "destiny", "act",
               "decision", "author", "agent", "instrument", "doll", "lever", "submission"],
    "magic": ["magic", "spell", "ritual", "witch", "sorcery", "alchemy", "demon", "spirit",
              "oracle", "charm", "talisman", "amulet", "divination", "ghost", "occult",
              "transformation", "vision", "dream", "wizard", "sorcerer", "enchantment",
              "prophecy", "omen", "curse", "miracle", "illusion", "rite", "magician"],
    "labor": ["labor", "work", "toil", "craft", "tool", "machine", "factory", "worker",
              "harvest", "weave", "forge", "mill", "trade", "guild", "wage", "artisan",
              "loom", "anvil", "plow", "spinner", "blacksmith", "carpenter", "mason",
              "potter", "miner", "weaver", "peasant", "servant", "drudgery", "skill"],
    "illich": ["conviviality", "vernacular", "tool", "institution", "commodity", "need",
               "dependence", "autonomy", "subsistence", "freedom"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="all", help="'all' (OMW langs), or comma list e.g. eng,fra,deu,jpn")
    ap.add_argument("--hyponyms", type=int, default=1, help="levels of hyponym expansion (breadth)")
    ap.add_argument("--max", type=int, default=0, help="cap total words (0 = no cap)")
    ap.add_argument("--out", default=os.path.join(HERE, "corpus", "words", "concepts.txt"))
    a = ap.parse_args()

    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find("corpora/" + pkg)
        except LookupError:
            nltk.download(pkg)
    from nltk.corpus import wordnet as wn

    langs = wn.langs() if a.langs == "all" else [s.strip() for s in a.langs.split(",")]
    print("[words] languages: %d (%s%s)" % (len(langs), ",".join(langs[:8]),
                                            " ..." if len(langs) > 8 else ""))

    def expand(synset, depth):
        pool = {synset}
        if depth > 0:
            for h in synset.hyponyms():
                pool |= expand(h, depth - 1)
        return pool

    words = set()
    for theme, seeds in SEEDS.items():
        for seed in seeds:
            syns = set()
            for s in wn.synsets(seed):
                syns |= expand(s, a.hyponyms)
            for s in syns:
                for lang in langs:
                    try:
                        for lemma in s.lemma_names(lang):
                            w = lemma.replace("_", " ").strip()
                            if w and 1 < len(w) <= 40:
                                words.add(w)
                    except Exception:
                        pass

    words = sorted(words)
    if a.max and len(words) > a.max:
        words = words[:a.max]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("# World of Shadow Work — concept vocabulary (auto-generated).\n")
        f.write("# Seeds: agency / magic / labor / Illich, expanded via WordNet + Open\n")
        f.write("# Multilingual WordNet. One word/phrase per line; '#' lines are skipped.\n")
        for w in words:
            f.write(w + "\n")
    print("[words] wrote %d concept words -> %s" % (len(words), a.out))


if __name__ == "__main__":
    main()
