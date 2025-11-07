# cli.py — Command Line Interface for Smart Text Assistant
# Handles user interaction, commands, and integrates the hybrid prediction system.

import sys
import shlex
import time 
import os
import pickle
import json
from random import choice

from hybrid_predictor import HybridPredictor
from config_manager import Config
from metrics_tracker import Metrics
from logger_utils import Log

# optional - logging config support from logging_config.yaml file
import logging.config, yaml, os
cfg_path = os.path.join(os.path.dirname(__file__), "utils", "logging_config.yaml")
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as fh:
        conf = yaml.safe_load(fh)
    logging.config.dictConfig(conf)
logger = logging.getLogger("intelligent_autocompleter")
logger.info("Logging configured from YAML")



# Constants and paths ------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_CORPUS = os.path.join(DATA_DIR, "demo_corpus.txt")
MODEL_PATH = os.path.join(DATA_DIR, "model_state.pkl")

BANNER = "Smart Text Assistant (type /help for commands)"

class CLI:
    def __init__(self):
        # Initialize logging first, so all other subsystems can use it
        log_path = os.path.join(DATA_DIR, "assistant.log")
        self.log = Log(log_path)
        self.log.info("=== Smart Text Assistant started ===")

        # Initialize core modules
        self.hp = HybridPredictor()
        self.cfg = Config()
        self.metrics = Metrics()
        self.running = True

        # Load existing model if present
        self._load_state()

    # === Main interactive loop ===
    def start(self):
        print(BANNER)
        self.log.info("CLI session started.")
        while self.running:
            try:
                line = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
                self.log.info("Session ended by user.")
                self._save_state()
                break

            if not line:
                continue

            if line.startswith("/"):
                self.cmd(line)
            else:
                t0 = time.perf_counter()
                self.hp.retrain(line)
                dt = time.perf_counter() - t0
                self.metrics.record("train_time", dt)
                self.log.debug(f"Retrained with line '{line}' in {dt:.3f}s.")
                print(f"Learnt: ({dt*1000:.1f} ms)")

    # Command processor ----------------------
    def cmd(self, line):
        p = shlex.split(line)
        if not p:
            return
        cmd = p[0].lower()

        match cmd:
            case "/q" | "/quit" | "/exit":
                self._save_state()
                self.running = False
                print("bye.")
                self.log.info("Exited normally.")
                return

            case "/help":
                print("Commands:")
                print("  /suggest <word>    : show predictions")
                print("  /mode <name>       : set model mode")
                print("  /train <file>      : train from file")
                print("  /bal <value>       : adjust balance")
                print("  /config [key val]  : view/set config")
                print("  /stats             : show performance metrics")
                print("  /bench             : benchmark suggestion speed")
                print("  /export <file>     : export data summary")
                print("  /quit              : exit program")
                return

            case "/suggest" if len(p) > 1:
                self._suggest(p[1])
                return

            case "/train" if len(p) > 1:
                self.train_file(p[1])
                return

            case "/mode" if len(p) > 1:
                self.set_mode(p[1])
                print("mode:", p[1])
                self.log.info(f"Mode changed to {p[1]}")
                return

            case "/bal" if len(p) > 1:
                try:
                    val = float(p[1])
                    self.hp.set_balance(val)
                    self.cfg.data["balance"] = val
                    self.cfg.save()
                    print("balance=", self.hp.alpha)
                    self.log.info(f"Balance set to {val}")
                except ValueError:
                    print("bad val")
                    self.log.warning(f"Invalid balance value: {p[1]}")
                return

            case "/config":
                self._config_cmd(p)
                return

            case "/stats":
                self.metrics.show()
                return

            case "/bench":
                self._bench()
                return

            case "/export" if len(p) > 1:
                self.export_data(p[1])
                return

            case _:
                print("unknown cmd")
                self.log.warning(f"Unknown command: {cmd}")

    # === Suggestion helper ===
    def _suggest(self, word):
        t0 = time.perf_counter()
        out = self.hp.suggest(word)
        dt = time.perf_counter() - t0
        self.metrics.record("suggest_time", dt)

        if out:
            for s, sc in out:
                print(f"{s}\t{sc:.3f}")
        else:
            print("(no suggestion)")
        self.log.debug(f"Suggested for '{word}' in {dt:.3f}s ({len(out) if out else 0} results).")

    # === Config helper ===
    def _config_cmd(self, p):
        if len(p) == 1:
            self.cfg.show()
        elif len(p) == 3:
            self.cfg.set(p[1], p[2])
            self.log.info(f"Config updated: {p[1]}={p[2]}")
        else:
            print("usage: /config [key val]")

    # === File-based training ===
    def train_file(self, path):
        try:
            with open(path, "r", encoding="utf8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            t0 = time.perf_counter()
            self.hp.train(lines)
            dt = time.perf_counter() - t0
            self.metrics.record("train_file_time", dt)
            print(f"trained on {len(lines)} lines in {dt:.2f}s")
            self._save_state()
            self.log.info(f"Trained from file {path} ({len(lines)} lines, {dt:.2f}s).")
        except Exception as e:
            print("Error:", e)
            self.log.error(f"Failed to train from {path}: {e}")

    # === Benchmark ===
    def _bench(self):
        words = ["the", "hello", "world", "this", "that", "there", "good"]
        t0 = time.perf_counter()
        for _ in range(100):
            self.hp.suggest(choice(words))
        dt = time.perf_counter() - t0
        avg = dt / 100
        print(f"bench: {avg:.5f}s avg per suggest")
        self.log.info(f"Benchmark completed: {avg:.5f}s avg")

    # === Data export ===
    def export_data(self, path):
        data = {
            "config": self.cfg.data,
            "metrics": {k: self.metrics.avg(k) for k in self.metrics.m},
        }
        try:
            with open(path, "w", encoding="utf8") as f:
                json.dump(data, f, indent=2)
            print("exported to", path)
            self.log.info(f"Exported summary to {path}")
        except Exception as e:
            print("export error:", e)
            self.log.error(f"Export failed: {e}")

    # === State management ===
    def _save_state(self):
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(self.hp, f)
            print("[saved model state]")
            self.log.info("Model state saved successfully.")
        except Exception as e:
            print("[warn] could not save model:", e)
            self.log.error(f"Model save failed: {e}")

    def _load_state(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.hp = pickle.load(f)
                print(f"[loaded existing model: {MODEL_PATH}]")
                self.log.info("Model loaded successfully.")
            except Exception as e:
                print("[warn] could not load model:", e)
                self.log.error(f"Model load failed: {e}")
        elif os.path.exists(DEFAULT_CORPUS):
            print(f"[AutoTrain] Loading corpus: {DEFAULT_CORPUS}")
            try:
                self.train_file(DEFAULT_CORPUS)
                print("[AutoTrain] Training complete.\n")
                self.log.info("Auto-trained from default corpus.")
            except Exception as e:
                print("[AutoTrain] Skipped due to error:", e)
                self.log.error(f"AutoTrain failed: {e}")
        else:
            print(f"[AutoTrain] No corpus found at {DEFAULT_CORPUS}")
            self.log.warning("No default corpus found for AutoTrain.")


if __name__ == "__main__":
    CLI().start()

