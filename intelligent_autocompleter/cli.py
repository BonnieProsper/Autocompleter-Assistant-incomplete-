# cli.py - CLI for smart text assistant

import sys, shlex, time
from hybrid_predictor import HybridPredictor
from config_manager import Config
from metrics_tracker import Metrics

BANNER = "Smart Text Assistant (type /help for cmds)"

class CLI:
    def __init__(self):
        self.hp = HybridPredictor()
        self.cfg = Config()
        self.metrics = Metrics()
        self.running = True

    def start(self):
        print(BANNER)
        while self.running:
            try:
                line = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
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
                print(f"Learnt: ({dt*1000:.1f} ms)")
    
    def cmd(self, line):
        p = shlex.split(line)
        if not p: return
        c = p[0].lower()
        
        if c in ("/q", "/quit", "/exit"):
            self.running = False
            print("bye.")
            return
            
        elif c == "/help":
            print("cmds: /suggest <word>, /mode <m>, /train <f>, /bal <v>")
            print("      /config [key val], /stats, /bench, /export <file>")
            return
            
        elif c == "/suggest" and len(p) > 1:
            w = p[1]
            t0 = time.perf_counter()
            out = self.hp.suggest(w)
            dt = time.perf_counter() - t0
            self.metrics.record("suggest_time", dt)
            for s, sc in out:
                print(f"{s}\t{sc:.3f}")
            return
                
        elif c == "/train" and len(p) > 1:
            self.train_file(p[1])
            return
        
        elif c == "/mode" and len(p) > 1:
            self.set_mode(p[1])
            print("mode:", p[1])
            return
            
        elif c == "/bal" and len(p) > 1:
            try:
                v = float(p[1])
                self.hp.set_balance(v)
                print("balance=", self.hp.alpha)
                self.cfg.data["balance"] = v
                self.cfg.save()
            except ValueError:
                print("bad val")
            return
        
        elif c == "/config":
            if len(p) == 1:
                self.cfg.show()
            elif len(p) == 3:
                self.cfg.set(p[1], p[2])
            else:
                print("usage: /config [key val]")
            return

        elif c == "/stats":
            self.metrics.show()
            return

        elif c == "/bench":
            self._bench()
            return

        elif c == "/export" and len(p) > 1:
            self.export_data(p[1])
            return

        else:
            print("unknown cmd")

    def train_file(self, path):
        import time
        try:
            with open(path, "r", encoding="utf8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            t0 = time.perf_counter()
            self.hp.train(lines)
            dt = time.perf_counter() - t0
            self.metrics.record("train_file_time", dt)
            print(f"trained on {len(lines)} lines in {dt:.2f}s")
        except Exception as e:
            print("err:", e)

    def _bench(self):
        from random import choice
        words = ["the", "hello", "world", "this", "that", "there", "good"]
        t0 = time.perf_counter()
        for _ in range(100):
            self.hp.suggest(choice(words))
        dt = time.perf_counter() - t0
        print(f"bench: {dt/100:.5f}s avg per suggest")

    def export_data(self, path):
        import json
        data = {
            "config": self.cfg.data,
            "metrics": {k: self.metrics.avg(k) for k in self.metrics.m},
        }
        try:
            with open(path, "w", encoding="utf8") as f:
                json.dump(data, f, indent=2)
            print("exported ->", path)
        except Exception as e:
            print("export err:", e)


if __name__ == "__main__":
    CLI().start()
