# evaluation/eval_fusion.py
# preset vs accuracy table using held-out split.

import random
from pathlib import Path
from collections import Counter
from intelligent_autocompleter.core.fusion_ranker import FusionRanker
from markov_predictor import MarkovPredictor

# helper to build markov on corpus and test next-word accuracy
CORPUS_PATH = Path(__file__).parent.parent / "data" / "demo_corpus.txt"
RANDOM_SEED = 42

def load_lines(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def split_lines(lines, hold_pct=0.2):
    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(lines)
    cut = int(len(lines) * (1 - hold_pct))
    return lines[:cut], lines[cut:]

def build_markov(lines):
    mk = MarkovPredictor()
    for ln in lines:
        mk.train_sentence(ln)
    return mk

def make_test_pairs(lines):
    pairs = []
    for ln in lines:
        toks = ln.split()
        if len(toks) >= 2:
            for i in range(len(toks)-1):
                pairs.append((toks[i].lower(), toks[i+1].lower()))
    return pairs

def evaluate_preset(preset, mk, pairs, topn=3):
    from embeddings import Embeddings
    # use empty embed/personal/freq maps, focused on markov maps
    ranker = FusionRanker(preset)
    ok = 0
    tot = 0
    for a, b in pairs:
        mk_scores = dict(mk.top_next(a, topn=topn*3))
        emb_scores = {}
        per = {}
        freq = {}
        results = ranker.rank(mk_scores, emb_scores, per, freq, topn=topn)
        preds = [w for w, _ in results]
        if b in preds:
            ok += 1
        tot += 1
    return ok, tot

def main():
    lines = load_lines(CORPUS_PATH)
    if not lines:
        print("No corpus at", CORPUS_PATH)
        return
    train, hold = split_lines(lines)
    mk = build_markov(train)
    pairs = make_test_pairs(hold)
    presets = ["balanced", "strict", "personal", "semantic"]
    print("preset, correct/total, accuracy")
    for p in presets:
        ok, tot = evaluate_preset(p, mk, pairs, topn=3)
        print(f"{p},{ok}/{tot},{ok/tot:.3f}")

if __name__ == "__main__":
    main()
