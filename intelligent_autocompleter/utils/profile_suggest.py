# check: put in different folder e.g tools?

# utils/profile_suggest.py
"""
Small profiling harness for HybridPredictor.suggest.
Usage:
  python tools/profile_suggest.py --warm 100 --iters 1000 --fragment "the quick brown"

Prints mean/median/std latency and basic sample of suggestions.
"""
import argparse
import time
import statistics

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

    hp = HybridPredictor(user="profile")
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

    print("Sample suggest output:", hp.suggest(args.fragment, topn=5))

if __name__ == "__main__":
    main()
