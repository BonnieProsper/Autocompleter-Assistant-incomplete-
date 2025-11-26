"""
cli.py - command line interface assistant
Features:
- Live autocompletion with color-coded predictions using HybridPredictor for predictions
- Adaptive learning, quiet feedback tracking and session persistence
- Plugin registry for custom expansion
- Real time optional weight readout
- Uses Rich for tables and formatting
"""

import os
import sys
import json
import time
import pickle
from typing import List, Tuple

# ui styling with Rich
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich import box

from intelligent_autocompleter.core.hybrid_predictor import HybridPredictor
from intelligent_autocompleter.core.adaptive_learner import AdaptiveLearner
from intelligent_autocompleter.core.feedback_tracker import FeedbackTracker
from intelligent_autocompleter.core.plugin_registry import PluginRegistry
from intelligent_autocompleter.utils.logger_utils import Log
from intelligent_autocompleter.utils.metrics_tracker import Metrics
from intelligent_autocompleter.utils.config_manager import Config
from intelligent_autocompleter.core.context_personal import CtxPersonal

""" check/include:
from intelligent_autocompleter.core.reasoner import ReasonerPipeline
from intelligent_autocompleter.core.semantic_engine import SemanticEngine

sem = SemanticEngine()  # or pass None during tests
pipeline = ReasonerPipeline(plugin_registry=None, semantic_engine=sem)

res = pipeline.analyze_input("gti commit -m 'fix'")
# res is a dict: res['corrections'], res['predictions'], res['semantic'], res['warnings'], ...
for k, v in res.items():
    print(k, v)
"""


# initialise console for rich output
console = Console()

# initialise file paths
ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
MODEL_PATH = os.path.join(DATA_DIR, "model_state.pkl")
SESSION_PATH = os.path.join(DATA_DIR, "session_state.json")
PLUGIN_CONFIG = os.path.join(DATA_DIR, "plugins_config.json")

class CLI:
    """Command-line interface (CLI) class to manage user interaction, autocompletion, and session persistence."""
    def __init__(self):
        """
        Initialize the CLI assistant:
        - Loads the HybridPredictor model
        - Sets up context (user-specific data) 
        - Initializes metrics tracking
        - Loads previous session data 
        """
        self.registry = PluginRegistry()
        self.hp = HybridPredictor(registry=self.registry)
        self.learner = ReinforcementLearner()
        
        self.context = CtxPersonal()
        self.metrics = Metrics()
        self.cfg = Config()
        
        self.session_data = []
        self.running = True
        self.show_feedback_logs = False  # toggleable
        os.makedirs(DATA_DIR, exist_ok=True)
        self._load_state()
        self._load_plugin_config()

    def run(self):
        """
        Main interactive loop of CLI:
        - Prompts the user for input.
        - Handles commands like /quit, /save.
        - Processes user input with autocompletion suggestions.
        """
        console.rule("[bold magenta]Intelligent Autocompleter[/bold magenta]")
        console.print("[cyan]Type sentences with live suggestions. Use #comment for notes.[/cyan]")
        console.print("Commands: /quit /weights /plugins /context /feedback /reset\n")

        # run loop as long as CLI is running
        while self.running:
            try:
                # prompt user for input (suggestions shown as they type)
                fragment = Prompt.ask("[green]You[/green]", default="")
                if not fragment: # if no input continue to next iteration
                    continue

                # handle commands
                if fragment.startswith("/"):
                    self._handle_command(fragment)
                    continue
                # handle inline comments
                if fragment.startswith("#"):
                    self._add_comment(fragment)
                    continue

                self._process_input(fragment)
            except (EOFError, KeyboardInterrupt):
                self._exit()
                break

    # COMMAND HANDLING -----------------------------------------------------------
    def _handle_command(self, cmd: str):
        """
        Handles special slash commands.
        """
        if cmd.startswith("/quit"):
            self._exit()
            return

        if cmd == "/weights":
            self._show_weights()
            return

        if cmd == "/plugins":
            self._show_plugins()
            return

        if cmd == "/context":
            self._show_context()
            return

        if cmd == "/feedback":
            self._show_feedback_log()
            return

        if cmd == "/reset":
            self._reset_state()
            return

        console.print(f"[red]Unknown command:[/red] {cmd}")

    # CORE INPUT PROCESSING ---------------------------------------------------------------
    def _process_input(self, fragmentment: str):
        """
        Loop:
        Prediction → display → user choice → apply → adaptive learning → feedback tracking
        """
        t0 = time.perf_counter()
        suggestions = self.hp.suggest(fragmentment) # get suggestions from HybridPredictor
        self.metrics.record("suggest_time", time.perf_counter() - t0) # record time for stats

        tokens = [t for t in fragment.strip().split() if t]
        if tokens:
            self.hp._learner.add_token(tokens[-1])
            
        if not suggestions:
            console.print("[dim](no suggestions)[/dim]")
            return

        self._display_suggestions(suggestions)
        # prompt user to select suggestion/custom word
        chosen = Prompt.ask("Pick # / override / Enter to skip", default="")
        if not chosen:
            return

        # accept suggestion using corresponding number
        if chosen.isdigit() and 1 <= int(chosen) <= len(suggestions):
            word, score, source = suggestions[int(chosen)-1]
            console.print(f"[green]Accepted:[/green] {word}  [dim]({source})[/dim]")

            self.hp.accept(word)
            self.context.learn(word)
            self.learner.update(word)
            self.feedback.push("accepted", {"word": word, "src": source})

            self.session_data.append({"input": fragmentment, "accepted": word})
            self._autosave()
            return

        # custom word
        word = chosen.strip()
        self.hp.retrain(word)
        self.context.learn(word)
        self.learner.update(word)
        self.feedback.push("custom", {"word": word})

        console.print(f"[cyan]Added custom:[/cyan] {word}")
        self.session_data.append({"input": fragmentment, "custom": word})
        self._autosave()

    # DISPLAY -------------------------------------------------------------------------------
    def _display_suggestions(self, suggestions: List[Tuple[str, float, str]]):
        """
        Display the autocompletion suggestions in a color-coded table.
        Each suggestion is ranked and color-coded based on its relevance, includes source.
        """
        table = Table(title="Predictions", box=box.SIMPLE, show_edge=False)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Word", style="bold")
        table.add_column("Score", justify="right", style="magenta")
        table.add_column("Source", justify="left", style="dim")

        for i, (w, score, src) in enumerate(suggestions[:5], 1):
            style = self._color_for_word(w)
            table.add_row(
                str(i),
                Text(w, style=style),
                f"{score:.3f}",
                src
            )
        console.print(table)

    def _color_for_word(self, word: str) -> str:
        """
        Assign a color to each word based on semantic categories.
        - Green for personal vocabulary (frequent words)
        - Cyan for short words, contextual
        - Magenta for title-case words (names etc, semantic)
        - Yellow as the fallback color (or mixed)
        """
        if word in self.context.freq:
            return "green"
        if len(word) <= 4:
            return "cyan"
        if word.istitle():
            return "magenta"
        return "yellow"

    # COMMENTS --------------------------------------------------------------
    def _add_comment(self, comment: str):
        """
        Handle comments entered by the user, for interactive note taking.
        Prints the comment in the console and adds it to session data.
        """
        note = comment.lstrip("#").strip()
        console.print(f"[dim italic]Comment:[/dim italic] {note}")
        self.session_data.append({"comment": note})
        self._autosave()


    # COMMAND: WEIGHTS/PLUGINS/CONTEXT/FEEDBACK --------------------------------------
    def _show_weights(self):
        panel = Panel(
            json.dumps(self.hp.get_weights(), indent=2),
            title="Current Weights",
            border_style="cyan"
        )
        console.print(panel)

        """or try:
            weights = self.hp.get_weights()
        except Exception:
            weights = {}
        panel = Panel.fit("\n".join([f"{k}: {v:.3f}" for k, v in weights.items()]), title="Live Weights", subtitle="ReinforcementLearner")
        console.print(panel)"""

    def _show_plugins(self):
        rows = []
        for name, obj in self.registry.plugins.items():
            enabled = "yes" if obj.enabled else "no"
            rows.append((name, enabled, obj.__class__.__name__))
        table = Table(title="Loaded Plugins", box=box.SIMPLE)
        table.add_column("Name")
        table.add_column("Enabled")
        table.add_column("Class")
        for r in rows:
            table.add_row(*r)

        console.print(table)

    def _show_context(self):
        freq_sorted = sorted(self.context.freq.items(), key=lambda kv: kv[1], reverse=True)
        table = Table(title="Personal Vocabulary", box=box.MINIMAL)
        table.add_column("Word")
        table.add_column("Freq")

        for w, c in freq_sorted[:20]:
            table.add_row(w, str(c))

        console.print(table)

    def _show_feedback_log(self):
        logs = self.feedback.dump()
        panel = Panel(
            json.dumps(logs, indent=2),
            title="Feedback Log",
            border_style="yellow"
        )
        console.print(panel)

    # STATE/SAVING/LOADING ----------------------------------------------------------
    def _autosave(self):
        """Automatically save session data every 5 entries."""
        if len(self.session_data) % 5 == 0:
            self._save_state(quiet=True)

    def _save_state(self, quiet=False):
        """
        Save the current model state and session data to disk.
        - Model state is saved with pickle.
        - Session state is saved as a JSON file.
        """
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(self.hp, f)
            with open(SESSION_PATH, "w", encoding="utf8") as f:
                json.dump(self.session_data, f, indent=2)
            if not quiet:
                console.print("[green]State saved.[/green]")
        except Exception as e:
            Log.write(f"[ERROR] Save: {e}")
            if not quiet:
                console.print(f"[red]Save failed:[/red] {e}")

    def _load_state(self):
        """
        Load the saved model and session state from disk.
        If no state exists then skip loading.
        """
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.hp = pickle.load(f)
                console.print("[dim]Model loaded.[/dim]")
            except Exception as e:
                Log.write(f"[ERROR] Load model: {e}")

        if os.path.exists(SESSION_PATH):
            try:
                with open(SESSION_PATH, "r", encoding="utf8") as f:
                    self.session_data = json.load(f)
                console.print(f"[dim]Restored {len(self.session_data)} entries.[/dim]")
            except Exception:
                pass

    def _load_plugin_config(self):
        """Load enable/disable flags."""
        if not os.path.exists(PLUGIN_CONFIG):
            return

        try:
            with open(PLUGIN_CONFIG, "r", encoding="utf8") as f:
                cfg = json.load(f)
            self.registry.apply_config(cfg)
        except Exception as e:
            console.print(f"[red]Plugin config load failed:[/red] {e}")

    # EXIT + RESET ------------------------------------------------------------------------
    def _reset_state(self):
        """Wipes session but not model."""
        self.session_data = []
        self.feedback.reset()
        console.print("[yellow]Session cleared.[/yellow]")

    def _exit(self):
        """
        Exit CLI program cleanly.
        Save current state.
        """
        console.rule("[red]Exiting[/red]")
        self._save_state()
        self.running = False


    def _session_summary(self):
        total_inputs = len([x for x in self.session_data if "input" in x])
        total_comments = len([x for x in self.session_data if "comment" in x])
        top_words = sorted(self.context.freq.items(), key=lambda kv: kv[1], reverse=True)[:5]
        t = Table(title="Session Summary", box=box.MINIMAL)
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="white")
        t.add_row("Total inputs", str(total_inputs))
        t.add_row("Comments added", str(total_comments))
        t.add_row("Top words", ", ".join(w for w, _ in top_words) or "(none)")
        return t


if __name__ == "__main__":
    CLI().run()

