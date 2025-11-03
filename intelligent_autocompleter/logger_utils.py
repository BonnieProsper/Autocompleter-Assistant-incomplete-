# logger_utils.py - logging and metrics    

import time
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "autocompleter.log")

class Log:
    @staticmethod
    def write(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    @staticmethod
    def metric(tag, value, unit=""):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {tag}: {value}{unit}"
        print(line)
        Log.write(line)

    @staticmethod
    def time_block(label):
        return _Timer(label)

class _Timer:
    def __init__(self, label):
        self.label = label
        self.start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        dur = round(time.time() - self.start, 3)
        Log.metric(f"{self.label} done", dur, "s")

