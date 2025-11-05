# metrics_tracker.py 

import time, json, os
from collections import defaultdict

class Metrics:
    def __init__(self, path="metrics.json"):
        self.path = path
        self.m = defaultdict(float)
        self.n = defaultdict(int)
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf8") as f:
                    d = json.load(f)
                for k, v in d.items():
                    self.m[k] = v["sum"]
                    self.n[k] = v["count"]
            except Exception:
                pass

    def save(self):
        d = {k: {"sum": self.m[k], "count": self.n[k]} for k in self.m}
        with open(self.path, "w", encoding="utf8") as f:
            json.dump(d, f, indent=2)

    def record(self, key, val):
        self.m[key] += val
        self.n[key] += 1
        self.save()

    def avg(self, key):
        if self.n[key] == 0: return 0.0
        return self.m[key] / self.n[key]

    def show(self):
        print("metrics:")
        for k in self.m:
            print(f"  {k:15} {self.avg(k):.4f}")

