"""
cli.py - command line interface assistant
Features:
- Live autocompletion with color-coded predictions
- Adaptive learning and session persistence
- Real time analytics summary
- Integrates hybrid_predictor to make predictions for the next word(s)
- Uses Rich for styling, session analytics, and adaptive assistance
"""

import os
import sys
import json
import time
import pickle

# ui styling with Rich
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich import box

from intelligent_autocompleter.core.hybrid_predictor import HybridPredictor
from intelligent_autocompleter.utils.logger_utils import Log
from intelligent_autocompleter.utils.metrics_tracker import Metrics
from intelligent_autocompleter.utils.config_manager import Config
from intelligent_autocompleter.core.context_personal import CtxPersonal

# initialise console for rich output
console = Console()

# initialise file paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(DATA_DIR, "model_state.pkl")
SESSION_PATH = os.path.join(DATA_DIR, "session_state.json")

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
        self.hp = HybridPredictor()
        self.context = CtxPersonal()
        self.metrics = Metrics()
        self.cfg = Config()
        self.session_data = []
        self.running = True
        self._load_state()

    def run(self):
        """
        Main interactive loop of CLI:
        - Prompts the user for input.
        - Handles commands like /quit, /save.
        - Processes user input with autocompletion suggestions.
        """
        console.rule("[bold magenta]Intelligent Autocompleter[/bold magenta]")
        console.print("[cyan]Type sentences with live suggestions. Use [yellow]#comment[/yellow] to add notes.[/cyan]")
        console.print("Press [bold]/quit[/bold] to exit.\n")

        # run loop as long as CLI is running
        while self.running:
            try:
                # prompt user for input (suggestions shown as they type)
                fragment = Prompt.ask("[green]You[/green]", default="")
                if not fragment: # if no input continue to next iteration
                    continue
                if fragment.startswith("/quit"):
                    self._exit()
                    break
                if fragment.startswith("/save"):
                    self._save_state()
                    continue
                # process users input with autocompletion
                self._process_input(fragment)
            except (EOFError, KeyboardInterrupt):
                self._exit()
                break

    # Core input processing + live suggestion handling --------------------------
    def process_input(self, fragment: str):
        """
        Process the user's input fragment:
        - Generate autocompletion suggestions
        - Accept user selection or allow custom input
        - Record session data and trigger autosave
        """
        start_time = time.perf_counter()

        # Handle inline comments
        if fragment.startswith("#"):
            self._add_comment(fragment)
            return

        # get autocompletion suggestions from HybridPredictor
        suggestions = self.hp.suggest(fragment)
        self.metrics.record("suggest_time", time.perf_counter() - start_time) # record time for stats
        if not suggestions:
            console.print("[dim](no suggestions)[/dim]")
            return
        self._display_suggestions(suggestions) # display colour coded table of suggestions

        # prompt user to select suggestion/custom word
        chosen = Prompt.ask("Pick number / type override / Enter to skip", default="")
        if not chosen:
            return

        # accept suggestion word corresponding to number
        if chosen.isdigit() and 1 <= int(chosen) <= len(suggestions):
            word, _ = suggestions[int(chosen) - 1] 
            console.print(f"[green]Accepted:[/green] {word}")
            self.hp.accept(word)
            self.context.learn(word) # learn for future predictions
            self.session_data.append({"input": fragment, "accepted": word})
        else:
            # custom word/retrain model
            custom = chosen.strip()
            self.hp.retrain(custom)
            console.print(f"[cyan]Added custom:[/cyan] {custom}")
            self.session_data.append({"input": fragment, "custom": custom})

        self._autosave() # autosave after processing input

    def display_suggestions(self, suggestions):
        """
        Display the autocompletion suggestions in a color-coded table.
        Each suggestion is ranked and color-coded based on its relevance.
        """
        table = Table(title="Predictions", box=box.SIMPLE, show_edge=False)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Word", style="bold")
        table.add_column("Score", justify="right", style="magenta")
        for i, (word, score) in enumerate(suggestions[:5], 1):
            style = self._color_for_word(word)
            table.add_row(str(i), Text(word, style=style), f"{score:.3f}")
        console.print(table)

    def color_for_word(self, word):
        """Assign a color to each word based on semantic categories.
        - Green for personal vocabulary (frequent words)
        - Cyan for short words
        - Magenta for title-case words (names, etc.)
        - Yellow as the fallback color
        """
        if word in self.context.freq:
            return "green"     # personal vocabulary
        if word.isalpha() and len(word) <= 4:
            return "cyan"      # short/contextual
        if word.istitle():
            return "magenta"   # semantic
        return "yellow"        # fallback or mixed

    def add_comment(self, comment):
        """
        Handle comments entered by the user, for interactive note taking.
        Prints the comment in the console and adds it to session data.
        """
        note = comment.lstrip("#").strip()
        console.print(f"[dim italic]Comment:[/dim italic] {note}")
        self.session_data.append({"comment": note})
        self._autosave()

    # Persistence: model + session memory---------------------------------------------
    def autosave(self):
        """Automatically save session data every 5 entries."""
        if len(self.session_data) % 5 == 0:
            self._save_state(quiet=True)

    def save_state(self, quiet=False):
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
            console.print(f"[red]Save failed:[/red] {e}")
            Log.write(f"[ERROR] Save: {e}")

    def load_state(self):
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
                console.print(f"[dim]Restored {len(self.session_data)} past entries.[/dim]")
            except Exception:
                pass

    def exit_session(self):
        """
        Exit CLI program cleanly.
        Save current state and print a session summary.
        """
        console.rule("[red]Exiting...[/red]")
        self._save_state()
        summary = self._session_summary()
        console.print(summary)
        self.running = False

    def session_summary(self):
        """
        Create and return a summary table (Rich table) of the current session. Including:
        - Total number of user inputs
        - Number of comments added
        - The user's top-used 5 most used words (from personal context learning)
        """
        total_inputs = len([x for x in self.session_data if "input" in x])
        total_comments = len([x for x in self.session_data if "comment" in x])
        top_words = sorted(self.context.freq.items(), key=lambda kv: kv[1], reverse=True)[:5]
        summary = Table(title="Session Summary", box=box.MINIMAL)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="white")
        summary.add_row("Total inputs", str(total_inputs))
        summary.add_row("Comments added", str(total_comments))
        summary.add_row("Top words", ", ".join(w for w, _ in top_words) or "(none)")
        return summary


if __name__ == "__main__":
    CLI().run()
