# check: put in different folder e.g tools?

# tools/profile_suggest.py
"""
Small profiling harness for HybridPredictor.suggest.
Usage:
  python tools/profile_suggest.py --warm 100 --iters 1000 --fragment "the quick brown"

Prints mean/median/std latency and basic sample of suggestions.
"""
import argparse
import time
import statistics
import random
from statistics import median
from pathlib import Path

try:
    from intelligent_autocompleter.core.hybrid_predictor import HybridPredictor
except Exception:
    from core.hybrid_predictor import HybridPredictor  # type: ignore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm", type=int, default=50, help="warmup iterations")
    parser.add_argument("--iters", type=int, default=500, help="measured iterations")
    parser.add_argument("--fragment", type=str, default="the quick brown", help="input fragment")
    args = parser.parse_args()

    hp = HybridPredictor(user="profile_user")
    # optionally you may want to pre-train with a large corpus to get realistic structures
    print("Warming up...")
    for _ in range(args.warm):
        _ = hp.suggest(args.fragment, topn=5)

    latencies = []
    print("Measuring...")
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _ = hp.suggest(args.fragment, topn=5)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    print("Stats (ms): mean=%.3f median=%.3f stdev=%.3f min=%.3f max=%.3f" % (
        statistics.mean(latencies),
        statistics.median(latencies),
        statistics.pstdev(latencies),
        min(latencies),
        max(latencies),
    ))

    warmup(hp)
    # compose queries (prefix fragments)
    queries = [
        "the quick",
        "please schedule",
        "thank you",
        "hello",
        "could you",
        "the",
        "project",
        "report",
    ]
    times = benchmark(hp, queries, iterations=300)
    s = summarize(times)
    print("Profiling summary (ms):", s)
    Path("tools/last_profile.json").write_text(str(s))
    print("Saved profile summary to tools/last_profile.json")

    print("Sample suggest output:", hp.suggest(args.fragment, topn=5))

# small synthetic dataset (or load a corpus file if you have one)
SAMPLE_SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "hello world this is a test sentence",
    "please schedule a meeting next monday at nine",
    "thank you for your contribution to the project",
    "could you show me the latest report",
]

def warmup(hp: HybridPredictor):
    for s in SAMPLE_SENTENCES:
        hp.retrain(s)

def benchmark(hp: HybridPredictor, queries, iterations=200):
    times = []
    for i in range(iterations):
        q = random.choice(queries)
        t0 = time.perf_counter()
        _ = hp.suggest(q, topn=6)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return times

def summarize(times):
    times_sorted = sorted(times)
    return {
        "count": len(times_sorted),
        "median_ms": median(times_sorted),
        "p90_ms": times_sorted[int(0.9 * len(times_sorted)) - 1],
        "max_ms": max(times_sorted),
    }

if __name__ == "__main__":
    main()
