# evaluation/eval_fusion.py
"""
Small evaluation harness for FusionRanker + HybridPredictor.

What it does (simple, reproducible):
 - Load text corpus (one sentence per line)
 - Build (prev_word -> next_word) test pairs
 - Train model on a train split, evaluate on held-out test pairs
 - For each preset in FusionRanker, compute:
     * Precision@k (k = 1,3,5)
     * Mean Reciprocal Rank (MRR)
 - Write a short markdown and JSON summary under evaluation/results/

to run use:
    python evaluation/eval_fusion.py
"""

from __future__ import annotations
import os
import json
import random
import time
from typing import List, Tuple, Dict
from collections import defaultdict

# import local modules (adjust if you moved files around)
# HybridPredictor should internally use the FusionRanker/presets or be easily configurable.
from hybrid_predictor import HybridPredictor
from fusion_ranker import FusionRanker
from logger_utils import Log

# -----------------------
# Config / paths
# -----------------------
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_PATH = os.path.join(ROOT, "data", "demo_corpus.txt")
OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Utilities / metrics
# -----------------------
def load_corpus(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus not found: {path}")
    lines = []
    with open(path, "r", encoding="utf8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return lines


def build_pairs(lines: List[str]) -> List[Tuple[str, str]]:
    """
    Return list of (prev_word, next_word) from sentence corpus.
    We generate one pair per adjacent token in each sentence.
    """
    pairs = []
    for ln in lines:
        toks = [t for t in ln.strip().split() if t]
        for a, b in zip(toks, toks[1:]):
            pairs.append((a.lower(), b.lower()))
    return pairs


def train_test_split(pairs: List[Tuple[str, str]], test_frac: float = 0.2, seed: int = 42):
    random.seed(seed)
    random.shuffle(pairs)
    ntest = max(1, int(len(pairs) * test_frac))
    test = pairs[:ntest]
    train = pairs[ntest:]
    return train, test


def precision_at_k(preds: List[str], gold: str, k: int) -> int:
    if not preds:
        return 0
    return 1 if gold in preds[:k] else 0


def reciprocal_rank(preds: List[str], gold: str) -> float:
    if not preds:
        return 0.0
    for i, p in enumerate(preds, start=1):
        if p == gold:
            return 1.0 / i
    return 0.0


# -----------------------
# Small experiment runner
# -----------------------
def evaluate_presets(
    hp_ctor,                 # callable that returns a fresh HybridPredictor instance
    presets: List[str],
    train_pairs: List[Tuple[str, str]],
    test_pairs: List[Tuple[str, str]],
    k_list=(1, 3, 5),
    topn=5,
) -> Dict[str, Dict]:
    """
    For each preset, create a FusionRanker with that preset and evaluate
    on the test pairs. We rely on HybridPredictor to expose suggest(...)
    that returns list of (word, score) pairs (or words).
    """
    results = {}
    for preset in presets:
        print(f"[eval] preset={preset} ...", end=" ", flush=True)
        # build model and set ranker preset
        hp = hp_ctor()
        # train with sentence-level training; we reconstruct sentences from train_pairs
        # simpler: collect sentences from pairs grouping by first token -> naive
        # but HybridPredictor has train(lines) so we need original lines. For simplicity,
        # we'll train hp by calling retrain on the joined pairs -> crude but works.
        for a, b in train_pairs:
            # feed as small sentence 'a b' to learn Markov counts
            hp.retrain(f"{a} {b}")

        # if HybridPredictor exposes FusionRanker or accepts preset, try to set it.
        # We'll try to set hp.ranker if present, else attempt to attach a ranker ourselves.
        try:
            # if HybridPredictor contains a ranker attribute, update it
            if hasattr(hp, "ranker"):
                hp.ranker.update_preset(preset)
            else:
                # try to create one and attach (best-effort)
                hp.ranker = FusionRanker(preset=preset, personalizer=hp.ctx if hasattr(hp, "ctx") else None)
        except Exception as e:
            Log.write(f"[warn] could not configure preset {preset}: {e}")

        # now evaluate
        tot = 0
        prec = {k: 0 for k in k_list}
        mrr_sum = 0.0
        for prev, gold in test_pairs:
            # ask model for suggestions for prev; model may expect a fragment; try both
            out = hp.suggest(prev, topn=topn)
            # hp.suggest commonly returns list of (word, score) or list of words.
            if not out:
                preds = []
            else:
                if isinstance(out[0], tuple):
                    preds = [w for w, _ in out]
                else:
                    preds = list(out)

            for k in k_list:
                prec[k] += precision_at_k(preds, gold, k)
            mrr_sum += reciprocal_rank(preds, gold)
            tot += 1

        # aggregate
        if tot == 0:
            print("no test examples!")
            results[preset] = {}
            continue

        prec_scores = {f"P@{k}": round(prec[k] / tot, 4) for k in k_list}
        mrr = round(mrr_sum / tot, 4)
        print(f"done (n={tot}) P@1={prec_scores['P@1']} MRR={mrr}")
        results[preset] = {"n": tot, "mrr": mrr, **prec_scores}
    return results


# -----------------------
# Report helpers
# -----------------------
def write_report(results: Dict[str, Dict], outdir: str):
    md = ["# FusionRanker Evaluation", f"Generated: {time.ctime()}", ""]
    md.append("| preset | n | P@1 | P@3 | P@5 | MRR |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for p, stats in results.items():
        if not stats:
            md.append(f"| {p} | - | - | - | - | - |")
            continue
        md.append(f"| {p} | {stats['n']} | {stats.get('P@1', '-')} | {stats.get('P@3', '-')} | {stats.get('P@5', '-')} | {stats.get('mrr', '-')} |")
    md_text = "\n".join(md)
    md_path = os.path.join(outdir, "fusion_eval.md")
    with open(md_path, "w", encoding="utf8") as fh:
        fh.write(md_text)
    json_path = os.path.join(outdir, "fusion_eval.json")
    with open(json_path, "w", encoding="utf8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[report] wrote {md_path} and {json_path}")


# -----------------------
# Main
# -----------------------
def main():
    try:
        lines = load_corpus(DATA_PATH)
    except FileNotFoundError as e:
        print("Corpus missing. Place demo_corpus.txt in data/ and rerun.")
        return

    if len(lines) < 5:
        print("Corpus too small for meaningful eval. Add more lines to data/demo_corpus.txt")
        return

    pairs = build_pairs(lines)
    # small quick run: 80/20 split
    train_pairs, test_pairs = train_test_split(pairs, test_frac=0.2)

    presets = ["strict", "balanced", "creative", "personal"]
    start = time.time()
    results = evaluate_presets(HybridPredictor, presets, train_pairs, test_pairs, k_list=(1, 3, 5), topn=5)
    duration = time.time() - start
    print(f"[eval] finished in {duration:.2f}s")

    write_report(results, OUT_DIR)


if __name__ == "__main__":
    main()
