# logger_utils.py -  for logging messages and performance metrics, timestamps etc

import time
import os
from datetime import datetime

# Directory where all log files will be stored
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)  # Create the folder if it doesn’t already exist

# Path to the default log file, can be overriden
DEFAULT_LOG_PATH = os.path.join(LOG_DIR, "autocompleter.log")

class Log:
    """Lightweight logger for writing messages and tracking metrics."""
    COLORS = {
        "DEBUG": "\033[90m",   # gray
        "INFO": "\033[94m",    # blue
        "WARNING": "\033[93m", # yellow
        "ERROR": "\033[91m",   # red
        "RESET": "\033[0m",
    }

    def __init__(self, path: str = None, use_color: bool = True):
        self.path = path or DEFAULT_LOG_PATH
        self.use_color = use_color

    def write(self, level:str, msg:str):
        """
        Append a log message to the log file with a timestamp.
        Each entry is written as: [YYYY-MM-DD HH:MM:SS] message
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {level:<7} | {msg}"
        
        # Write to the log file
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        # print to console (color enabled etc)
        if self.use_color and level in self.COLORS:
            print(f"{self.COLORS[level]}{line}{self.COLORS['RESET']}")
        else:
            print(line)

    # Public logging methods
    def debug(self, msg: str):
        self._write("DEBUG", msg)

    def info(self, msg: str):
        self._write("INFO", msg)

    def warning(self, msg: str):
        self._write("WARNING", msg)

    def error(self, msg: str):
        self._write("ERROR", msg)
    
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
        with open(DEFAULT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def time_block(label):
        """
        Helper for measuring execution time of a code block.
        To use:
            with Log.time_block("data_processing"):
                do_some_work()
        It automatically logs how long the block took.
        """
        return _Timer(label)


class _Timer:
    """Context manager used internally to measure time for a code block."""
    def __init__(self, label):
        self.label = label
        self.start = time.time()  # record the start time when the block begins

    def __enter__(self):
        # nothing special to do on enter, just return self
        return self

    def __exit__(self, exc_type, exc, tb):
        """When exiting the 'with' block, calculate how long it took and record it as a metric. """
        dur = round(time.time() - self.start, 3)  # duration in seconds (rounded)
        Log.metric(f"{self.label} done", dur, "s")


