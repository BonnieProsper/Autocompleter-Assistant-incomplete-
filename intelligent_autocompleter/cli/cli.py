# cli.py — command line interface for user interaction, commands, 
# integrates the hybrid prediction system,
# uses Rich for styling, session analytics, and adaptive assistance

import os
import sys 
import shlex
import time
import pickle
import json
from random import choice
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich import box

# ui styling with Rich
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich import box

from hybrid_predictor import HybridPredictor
from config_manager import Config
from metrics_tracker import Metrics
from logger_utils import Log

# Paths setup ----------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "model_state.pkl")
DEFAULT_CORPUS = os.path.join(DATA_DIR, "demo_corpus.txt")

console = Console()
BANNER = "[bold cyan]Smart Text Assistant[/bold cyan] — type [yellow]/help[/yellow] for commands."

# Command Line Interface logic ---------------------------------
class CLI:
    def __init__(self):
        self.hp = HybridPredictor()
        self.cfg = Config()
        self.metrics = Metrics()
        self.log = Log
        self.session_active = False
        self.session_buf: List[str] = []
        self.history: List[str] = []
        self._load_state(quiet=True)
        self.log.write("CLI initialized")

    # main REPL loop -------------------------------
    # waits for user input then responds
    def start(self):
        console.rule("[bold magenta]Smart Text Assistant[/bold magenta]")
        console.print(BANNER)
        while True:
            try:
                line = Prompt.ask("[green]>>[/green]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[red]Exiting gracefully...[/red]")
                self._save_state()
                break

            if not line:
                continue # ignore empty input
            self.history.append(line)

            if line.startswith("/"): # all commands start with '/'
                self._handle_command(line)
            else:
                self._train_line(line)

    # training line fed into model by user, for learning purposes (not demo corpus)
    def _train_line(self, text: str):
        t0 = time.perf_counter()
        self.hp.retrain(text)
        dt = time.perf_counter() - t0
        self.metrics.record("train_time", dt)
        if self.session_active:
            self.session_buf.append(text)
        console.print(f"[magenta]Learned in {dt*1000:.1f} ms[/magenta]")

    # interpret and execute commands
    def _handle_command(self, raw: str):
        parts = shlex.split(raw)
        if not parts:
            return
        cmd = parts[0].lower()

        # match commands, including possible shortcuts
        match cmd:
            case "/q" | "/quit" | "/exit":
                self._save_state()
                console.print("[bold red]Bye![/bold red]")
                sys.exit(0)

            case "/help":
                self._show_help()

            case "/suggest":
                if len(parts) < 2:
                    console.print("Usage: /suggest <fragment>")
                    return
                self._suggest(" ".join(parts[1:]))

            case "/compose":
                self._compose_mode()

            case "/train":
                if len(parts) < 2:
                    console.print("Usage: /train <file>")
                    return
                self._train_file(parts[1])

            case "/session":
                self._handle_session(parts[1:])

            case "/config":
                self._handle_config(parts[1:])

            case "/stats":
                self.metrics.show()

            case "/bench":
                self._bench()

            case "/recall":
                self._recall_last()

            case "/save":
                self._save_state()

            case "/load":
                self._load_state()

            case _:
                console.print(f"[red]Unknown command:[/red] {cmd}")

    # generate suggestions for text 
    def _suggest(self, frag: str):
        t0 = time.perf_counter()
        suggestions = self.hp.suggest(frag)
        dt = time.perf_counter() - t0
        self.metrics.record("suggest_time", dt)

        if not suggestions:
            console.print("[dim](no suggestions)[/dim]")
            return

        # Rich table for nicer output (tbd)
        table = Table(title="Suggestions", box=box.SIMPLE, show_edge=False)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Word", style="bold")
        table.add_column("Score", justify="right", style="magenta")
        for i, (word, score) in enumerate(suggestions[:5], 1):
            table.add_row(str(i), word, f"{score:.3f}")
        console.print(table)

        # user can accept/add own word
        choice_input = Prompt.ask("Pick number / type word / Enter to skip", default="")
        if not choice_input:
            return
        if choice_input.isdigit():
            # accept predicted suggestion
            idx = int(choice_input)
            if 1 <= idx <= len(suggestions):
                chosen = suggestions[idx - 1][0]
                self.hp.accept(chosen)
                self._save_state()
                console.print(f"[green]Accepted:[/green] {chosen}")
        else:
            # user adds word
            word = choice_input.strip().lower()
            self.hp.retrain(word)
            self._save_state()
            console.print(f"[cyan]Added:[/cyan] {word}")

    # interactive mode, live hints while typing
    def _compose_mode(self):
        console.rule("[yellow]Compose Mode[/yellow]")
        buf = []
        while True:
            raw = Prompt.ask("[blue]Compose[/blue]", default="").strip()
            if raw in ("/done", "/exit"):
                break
            # quick accept best suggestion for last word
            if raw == "/accept" and buf:
                last = buf[-1]
                suggs = self.hp.suggest(last)
                if suggs:
                    chosen = suggs[0][0]
                    buf[-1] = chosen
                    self.hp.retrain(chosen)
                    console.print(f"[green]Accepted:[/green] {chosen}")
                continue
            if not raw:
                continue

            """ To include??:
            if len(buf) > 1:
            prev = buf[-2]
            suggs = self.hp.suggest(prev)
            if suggs:
                top = suggs[0][0]
                # If user types smth other than top suggestion treat as reject
                if tok.lower() != top.lower():
                    self.hp.feedback.record(context=prev, suggestion=top, accepted=False)
                    self.hp.feedback.save()

            """
                
            # update sentence buffer
            buf.extend(raw.split())
            # display live hints
            last = buf[-1]
            hints = self.hp.suggest(last)
            if hints:
                hint_text = ", ".join(f"{w} ({s:.2f})" for w, s in hints[:3])
                console.print(f"[cyan][hint][/cyan] {hint_text}")

        # save final text
        sentence = " ".join(buf)
        if sentence:
            self.hp.retrain(sentence)
            if self.session_active:
                self.session_buf.append(sentence)
            console.print(f"[magenta]Saved sentence:[/magenta] {sentence}")
        console.rule("[dim]Compose Ended[/dim]")

    # train model using text file (e.g demo_corpus.txt)
    def _train_file(self, path: str):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            console.print(f"[red]File not found:[/red] {path}")
            return
        with open(path, "r", encoding="utf8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        t0 = time.perf_counter()
        self.hp.train(lines)
        dt = time.perf_counter() - t0
        self.metrics.record("train_file_time", dt)
        self._save_state()
        console.print(f"[green]Trained on {len(lines)} lines in {dt:.2f}s[/green]")

    # Session management ----------------------------------
    # start, end, save ongoing learning sessions
    def _handle_session(self, args: List[str]):
        if not args:
            console.print("Usage: /session start|end|save <file>")
            return
        sub = args[0].lower()
        match sub:
            case "start":
                self.session_active = True
                self.session_buf = []
                console.print("[green]Session started[/green]")
            case "end":
                if not self.session_active:
                    console.print("[yellow]No active session[/yellow]")
                    return
                self.session_active = False
                n = len(self.session_buf)
                console.print(f"[cyan]Session ended ({n} entries)[/cyan]")
                if n > 0:
                    self.hp.train(self.session_buf)
                    self._save_state()
                    self.session_buf = []
            case "save":
                if len(args) < 2:
                    console.print("Usage: /session save <file>")
                    return
                self._session_save(args[1])

    # save session buffer to file
    def _session_save(self, path: str):
        try:
            with open(path, "w", encoding="utf8") as fh:
                for s in self.session_buf:
                    fh.write(s + "\n")
            console.print(f"[green]Session saved -> {path}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to save session:[/red] {e}")

    # Configuration handler --------------------------------------
    # view and change settings
    def _handle_config(self, args: List[str]):
        if not args:
            self.cfg.show()
            return
        if len(args) == 2:
            k, v = args
            self.cfg.set(k, v)
            self.cfg.save()
            console.print(f"[cyan]Config updated:[/cyan] {k}={v}")
        else:
            console.print("Usage: /config [key val]")

    # Utility commands -----------------------------------------------
    # recall, benchmark, stats
    def _recall_last(self):
        if not self.history:
            console.print("[dim]No previous input[/dim]")
            return
        last = self.history[-1]
        console.print(f"[yellow]Last command:[/yellow] {last}")

    # benchmark to test suggestion speed
    def _bench(self):
        words = ["the", "this", "hello", "assistant", "python", "good"]
        t0 = time.perf_counter()
        for _ in range(100):
            self.hp.suggest(choice(words))
        dt = time.perf_counter() - t0
        console.print(f"[cyan]Benchmark avg: {dt/100:.5f}s per suggest[/cyan]")

    # Persistence ----------------------------------------------------------
    # to save and load model state
    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(self.hp, f)
            console.print("[green]Model saved.[/green]")
        except Exception as e:
            console.print(f"[red]Save failed:[/red] {e}")
            self.log.write(f"Save failed: {e}")

    def _load_state(self, quiet: bool = False):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.hp = pickle.load(f)
                if not quiet:
                    console.print("[green]Loaded existing model.[/green]")
            except Exception as e:
                console.print(f"[red]Load failed:[/red] {e}")
                self.log.write(f"Load failed: {e}")
        elif os.path.exists(DEFAULT_CORPUS):
            console.print(f"[yellow]Autotrain: using default corpus...[/yellow]")
            self._train_file(DEFAULT_CORPUS)
        else:
            console.print("[dim]No saved model or corpus found.[/dim]")

    # Help Display -----------------------------------------------------
    # list of all commands
    def _show_help(self):
        help_text = Table(title="Available Commands", box=box.MINIMAL_DOUBLE_HEAD)
        help_text.add_column("Command", style="cyan", no_wrap=True)
        help_text.add_column("Description", style="white")
        help_text.add_row("/suggest <frag>", "Show predictions for a fragment")
        help_text.add_row("/compose", "Enter interactive writing mode")
        help_text.add_row("/train <file>", "Train model on file")
        help_text.add_row("/session start|end|save", "Manage user sessions")
        help_text.add_row("/config [key val]", "View or set configuration")
        help_text.add_row("/recall", "Show last entered command")
        help_text.add_row("/bench", "Run performance benchmark")
        help_text.add_row("/stats", "Display runtime metrics")
        help_text.add_row("/quit", "Exit the program")
        console.print(help_text)

if __name__ == "__main__":
    CLI().start()
