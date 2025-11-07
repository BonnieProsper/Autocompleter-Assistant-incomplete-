# cli.py — command line interface for user interaction, commands, integrates the hybrid prediction system.

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

# logging config support from logging_config.yaml file, is optional
import logging.config, yaml, os
cfg_path = os.path.join(os.path.dirname(__file__), "utils", "logging_config.yaml")
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as fh:
        conf = yaml.safe_load(fh)
    logging.config.dictConfig(conf)
logger = logging.getLogger("intelligent_autocompleter")
logger.info("Logging configured from YAML")



# Initialise constants/paths --------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_CORPUS = os.path.join(DATA_DIR, "demo_corpus.txt")
MODEL_PATH = os.path.join(DATA_DIR, "model_state.pkl")

BANNER = "Smart Text Assistant (type /help for commands)"

class CLI:
    def __init__(self):
        # Initialize logging first, so all other subsystems can use it
        self.log = Log
        self.log.info("=== Smart Text Assistant started ===")

        # Initialize core modules
        self.hp = HybridPredictor()
        self.cfg = Config()
        self.metrics = Metrics()
        self.running = True

        # Load existing model if present
        self._load_state()

    # Main interactive loop -----------------
    def start(self):
        print(BANNER)
        self.log.write("CLI session started.")
        while self.running:
            try:
                line = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
                self.log.write("Session ended by user.")
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

        if cmd in ("/q", "/quit", "/exit"):
            self._save_state()
            self.running = False
            print("bye.")
            self.log.write("Exited normally.")
            return

        if cmd == "/help":
            print("Commands:")
            print("  /suggest <word>    : show predictions")
            print("  /train <file>      : train from file")
            print("  /bal <value>       : adjust balance")
            print("  /config [key val]  : view/set config")
            print("  /stats             : show performance metrics")
            print("  /bench             : benchmark suggestion speed")
            print("  /export <file>     : export data summary")
            print("  /quit              : exit program")
            return

        if cmd == "/suggest" and len(p) > 1:
            self._suggest(p[1])
            return

        if cmd == "/train" and len(p) > 1:
            self.train_file(p[1])
            return

        if cmd == "/bal" and len(p) > 1:
            try:
                val = float(p[1])
                self.hp.set_balance(val)
                self.cfg.data["balance"] = val
                self.cfg.save()
                print("balance=", self.hp.alpha)
                self.log.write(f"Balance set to {val}")
            except ValueError:
                print("bad val")
                self.log.write(f"Invalid balance value: {p[1]}")
            return

        if cmd == "/config":
            if len(p) == 1:
                self.cfg.show()
            elif len(p) == 3:
                self.cfg.set(p[1], p[2])
                self.log.write(f"Config updated: {p[1]}={p[2]}")
            else:
                print("usage: /config [key val]")
            return

        if cmd == "/stats":
            self.metrics.show()
            return

        if cmd == "/bench":
            self._bench()
            return

        if cmd == "/export" and len(p) > 1:
            self.export_data(p[1])
            return

        print("unknown cmd")
        self.log.write(f"Unknown command: {cmd}")

    # Suggestion helper -----------------
    def _suggest(self, word):
        t0 = time.perf_counter()
        out = self.hp.suggest(word)
        dt = time.perf_counter() - t0
        self.metrics.record("suggest_time", dt)
        if not out:
            print("(no suggestion)")
            self.log.write(f"No suggestions for '{word}'")
            return

        # give numbered choices and allow user to accept them
        print("\nSuggestions:")
        for i, (s, sc) in enumerate(out, start=1):
            print(f"  {i}. {s} (score {sc:.3f})")
        print("  0. none / type another word")

        choice = input("Pick [num / word / Enter]: ").strip()
        if not choice:
            return
        if choice.isdigit():
            idx = int(choice)
            if idx == 0:
                return
            if 1 <= idx <= len(out):
                chosen = out[idx - 1][0]
                self.hp.accept(chosen)
                self._save_state()
                print(f"Accepted '{chosen}'")
                self.log.write(f"User accepted suggestion '{chosen}'")
                return
            else:
                print("bad number")
                return
        # typed word selection
        chosen = choice.lower().strip()
        # if typed matches suggestion accept it
        if any(chosen == s for s, _ in out):
            self.hp.accept(chosen)
            self._save_state()
            print(f"Accepted '{chosen}'")
            self.log.write(f"User accepted suggestion '{chosen}'")
        else:
            # treat as new word, retrain/add it
            self.hp.retrain(chosen)
            self._save_state()
            print(f"Added '{chosen}' to model")
            self.log.write(f"User added new word '{chosen}'")

    # Train using file -----------------
    def train_file(self, path):
        try:
            # expand paths
            path = os.path.expanduser(path)
            with open(path, "r", encoding="utf8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            t0 = time.perf_counter()
            self.hp.train(lines)
            dt = time.perf_counter() - t0
            self.metrics.record("train_file_time", dt)
            print(f"trained on {len(lines)} lines in {dt:.2f}s")
            self._save_state()
            self.log.write(f"Trained from file {path} ({len(lines)} lines, {dt:.2f}s)")
        except Exception as e:
            print("Error:", e)
            self.log.write(f"Failed to train from {path}: {e}")

    # Benchmark -------------------------------
    def _bench(self):
        words = ["the", "hello", "world", "this", "that", "there", "good"]
        t0 = time.perf_counter()
        for _ in range(100):
            self.hp.suggest(choice(words))
        dt = time.perf_counter() - t0
        avg = dt / 100
        print(f"bench: {avg:.5f}s avg per suggest")
        self.log.info(f"Benchmark completed: {avg:.5f}s avg")

    # export data ----------------------------
    def export_data(self, path):
        data = {
            "config": self.cfg.data,
            "metrics": {k: self.metrics.avg(k) for k in self.metrics.m},
        }
        try:
            with open(path, "w", encoding="utf8") as f:
                json.dump(data, f, indent=2)
            print("exported to", path)
            self.log.write(f"Exported summary to {path}")
        except Exception as e:
            print("export error:", e)
            self.log.error(f"Export failed: {e}")

    # State management -------------------------------------
    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(self.hp, f)
            print("[saved model state]")
            self.log.write("Model state saved successfully.")
        except Exception as e:
            print("[warn] could not save model:", e)
            self.log.write(f"Model save failed: {e}")

    def _load_state(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.hp = pickle.load(f)
                print(f"[loaded existing model: {MODEL_PATH}]")
                self.log.info("Model loaded successfully.")
            except Exception as e:
                print("[warn] could not load model:", e)
                self.log.write(f"Model load failed: {e}")
        elif os.path.exists(DEFAULT_CORPUS):
            print(f"[AutoTrain] Loading corpus: {DEFAULT_CORPUS}")
            try:
                self.train_file(DEFAULT_CORPUS)
                print("[AutoTrain] Training complete.\n")
                self.log.write("Auto-trained from default corpus.")
            except Exception as e:
                print("[AutoTrain] Skipped due to error:", e)
                self.log.write(f"AutoTrain failed: {e}")
        else:
            print(f"[AutoTrain] No corpus found at {DEFAULT_CORPUS}")
            self.log.write("No default corpus found for AutoTrain.")


if __name__ == "__main__":
    CLI().start()

