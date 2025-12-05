# bench_profiling.py - check: put in different folder
"""
Simple profiling harness for HybridPredictor.suggest

Usage:
    python -m bench_profiling --runs 200 --warmup 20
"""

import time
import statistics
import argparse
from intelligent_autocompleter.autocompleter import AutoCompleter


def profile(ac: AutoCompleter, fragments: list[str], runs: int = 200, warmup: int = 20):
    # warmup
    for _ in range(warmup):
        ac.suggest(fragments[_ % len(fragments)], topn=6)

    times = []
    for i in range(runs):
        f = fragments[i % len(fragments)]
        t0 = time.time()
        ac.suggest(f, topn=6)
        t = (time.time() - t0) * 1000.0
        times.append(t)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    ac = AutoCompleter()
    try:
        # lightweight synthetic set
        frags = [
            "hello wor",
            "this is a t",
            "import nump",
            "how to bui",
            "openai chatg",
        ]
        times = profile(ac, frags, runs=args.runs, warmup=args.warmup)
        print("calls:", len(times))
        print("mean ms:", statistics.mean(times))
        print("median ms:", statistics.median(times))
        print("p99 ms:", sorted(times)[int(len(times) * 0.99) - 1])
    finally:
        # ensure clean shutdown (persist)
        pass


if __name__ == "__main__":
    main()
