# profiling/profile.py
"""
Simple/small profiling harness to measure suggest latency.
Usage:
    python -m profiling.profile
"""
import time
from pathlib import Path
from intelligent_autocompleter.core.hybrid_predictor import HybridPredictor


def bench(predictor, fragment="hello", runs=1000):
    t0 = time.time()
    for _ in range(runs):
        predictor.suggest(fragment, topn=5)
    dur = time.time() - t0
    return dur / runs


def main():
    hp = HybridPredictor()
    # warmup
    hp.train(["hello world", "hello there", "say hello world again"])
    hp.suggest("hello", topn=5)
    print("Warmup done")
    avg = bench(hp, "hello", runs=500)
    print(f"avg latency/suggest: {avg:.6f}s")


if __name__ == "__main__":
    main()
