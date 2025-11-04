# logger_utils.py -  for logging messages and performance metrics

import time
import os
from datetime import datetime

# Directory where all log files will be stored
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)  # Create the folder if it doesn’t already exist

# Path to the main log file
LOG_PATH = os.path.join(LOG_DIR, "autocompleter.log")

class Log:
    """Lightweight logger for writing messages and tracking metrics."""
    
    @staticmethod
    def write(msg):
        """
        Append a log message to the log file with a timestamp.
        Each entry is written as: [YYYY-MM-DD HH:MM:SS] message
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    @staticmethod
    def metric(tag, value, unit=""):
        """
        Record a metric (like timing, counts, or performance stats).
        Prints to the console and also logs it to the file.
        Example: [12:45:02] API call latency: 0.123s
        """
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {tag}: {value}{unit}"
        print(line)
        Log.write(line)

    @staticmethod
    def time_block(label):
        """
        Helper for measuring execution time of a code block.
        Use it like:
            with Log.time_block("data_processing"):
                do_some_work()
        It will automatically log how long the block took.
        """
        return _Timer(label)


class _Timer:
    """Context manager used internally to measure time for a code block."""

    def __init__(self, label):
        self.label = label
        self.start = time.time()  # Record the start time when the block begins

    def __enter__(self):
        # Nothing special to do on enter, just return self
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        When exiting the 'with' block, calculate how long it took
        and record it as a metric.
        """
        dur = round(time.time() - self.start, 3)  # Duration in seconds (rounded)
        Log.metric(f"{self.label} done", dur, "s")


