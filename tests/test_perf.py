# test_perf.py - rough perf check
import time
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hybrid_predictor import HybridPredictor

hp = HybridPredictor("bench")

sample = [
    "python is great for data work",
    "java and c++ are fast but verbose",
    "machine learning needs lots of data",
]
hp.train(sample)

words = ["python", "data", "fast", "machine", "work"]
times = []
for i in range(30):
    w = random.choice(words)
    t0 = time.time()
    _ = hp.suggest(w)
    times.append(time.time()-t0)

avg = sum(times)/len(times)
print(f"avg latency: {round(avg*1000,2)} ms over {len(times)} runs")
