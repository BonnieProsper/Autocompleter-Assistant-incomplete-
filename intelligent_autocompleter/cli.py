# cli.py - CLI for smart text assistant

import sys, shlex
from hybrid_predictor import HybridPredictor

BANNER = "Smart Text Assistant (type /help for cmds)"

class CLI:
    def __init__(self):
        self.hp = HybridPredictor()
        self.mode = "hybrid"
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
                self.hp.retrain(line)
                print("learned.")
    
    def cmd(self, line):
        p = shlex.split(line)
        if not p: return
        c = p[0].lower()
        if c in ("/q", "/quit", "/exit"):
            self.running = False
            print("bye.")
        elif c == "/help":
            print("cmds: /suggest <word>, /mode <m>, /train <f>, /bal <v>, /q")
        elif c == "/suggest" and len(p) > 1:
            w = p[1]
            out = self.hp.suggest(w)
            for s, sc in out:
                print(f"{s}\t{sc:.3f}")
        elif c == "/train" and len(p) > 1:
            self.train_file(p[1])
        elif c == "/mode" and len(p) > 1:
            self.set_mode(p[1])
        elif c == "/bal" and len(p) > 1:
            try:
                v = float(p[1])
                self.hp.set_balance(v)
                print("balance=", self.hp.alpha)
            except ValueError:
                print("bad val")
        else:
            print("unknown cmd")

    def train_file(self, path):
        try:
            with open(path, "r", encoding="utf8") as f:
                txt = [ln.strip() for ln in f if ln.strip()]
            self.hp.train(txt)
            print(f"trained on {len(txt)} lines.")
        except Exception as e:
            print("err:", e)

    def set_mode(self, m):
        m = m.lower()
        if m not in ("markov", "embed", "hybrid"):
            print("bad mode.")
            return
        self.mode = m
        print("mode ->", m)


if __name__ == "__main__":
    CLI().start()
