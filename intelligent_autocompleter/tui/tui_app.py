# tui_app.py — Autocompleter Assistant TUI Application
# -------------------------------------------------------
# Text based terminal UI that wraps the autocompleter stack into a single cohesive and interactive app.
# Features:
#  - Live predictions as you type
#  - Color-coded suggestion list with model hints
#  - Accept suggestions using TAB or 1–5 keys
#  - Real-time latency + autosave indicators
#  - Session persistence + inline context learning
# If the predictor learns from your choices, the UI shows it.
# If the system hesitates, the latency counter tells you why.
# -------------------------------------------------------

from __future__ import annotations
import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Input, Static, DataTable, LoadingIndicator
)
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual import events

from intelligent_autocompleter.core.hybrid_predictor import HybridPredictor
from intelligent_autocompleter.core.context_personal import CtxPersonal
from intelligent_autocompleter.utils.logger_utils import Log

# store session text/learned model
SESSION_PATH = Path("data/session_tui.json")
MODEL_PATH = Path("data/model_state.pkl")


class SuggestionPanel(Static):
    """
    Right-side suggestion panel.
    Displays up to 5 predictions, showing:
     - index shortcuts (1–5)
     - color-coded confidence scores
     - the predicted word
    """
    def update_predictions(self, predictions):
        lines = []
        # placeholder
        self.update("")
        if not predictions:
            self.update("[dim]No suggestions[/dim]")
            return
        
        for i, (word, score) in enumerate(predictions[:5], 1):
            # Confidence score colours:
            # > 0.7: High confidence = Green
            # > 0.4: Medium confidence = Cyan
            # Else: Low confidence = Yellow
            color = "green" if score > 0.7 else "cyan" if score > 0.4 else "yellow"
            lines.append(f"[b]{i}[/b] • [{color}]{word}[/{color}]  [dim]{score:.3f}[/dim]")
        self.update("\n".join(lines))


class ContextView(Static):
    """
    Shows what the local learner (CtxPersonal) has learnt,
    which helps the user understand how their choices shape future predictions.
    """
    def update_context(self, context: CtxPersonal):
        if not context.freq:
            self.update("[dim]No learned tokens yet[/dim]")
            return
        # 6 most frequently accepted words
        top_items = sorted(context.freq.items(), key=lambda kv: kv[1], reverse=True)[:6]
        formatted = ", ".join(f"{w}({n})" for w, n in top_items)
        self.update(f"[b]Top learned words:[/b] {formatted}")


class TypingLatency(Static):
    """
    Bottom-left readout showing how long the last prediction took.
    """
    def set_latency(self, seconds: float):
        ms = round(seconds * 1000)
        self.update(f"[dim]Latency:[/dim] {ms}ms")

# Main Application -----------------------------------------------------------------
class TUIAutocompleter(App):
    """
    The main Textual app. 
    Manages:
     - user input
     - prediction updates
     - personal context learning
     - persistence to disk
    Architecture:
     - UI events to prediction engine
     - prediction engine to reactive state
     - reactive state to UI updates
    """
    CSS_PATH = "tui_style.css"
  
    # keyboard shortcuts for user
    BINDINGS = [
        ("tab", "accept_top", "Accept Top Suggestion"),
        ("ctrl+s", "save_session", "Save Session"),
        ("ctrl+r", "reset_context", "Reset"),
    ]

    # reactive values that automatically refresh widgets when changed
    user_text = reactive("") # whats currently in input box
    suggestions = reactive([]) # most recent prediction results
    latency = reactive(0.0) # measured model inference time

    def __init__(self):
        super().__init__()
        self.hp = HybridPredictor()
        self.context = CtxPersonal()
        self.load_state()

    # UI --------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(): # Main split view
            with Container(id="left"):
                yield Input(placeholder="Start typing…", id="text_input")
                yield ContextView(id="context")

            with Container(id="right"):
                yield SuggestionPanel(id="predictions")

        with Horizontal(id="bottom"):
            yield TypingLatency(id="latency")
            yield Static(id="status")
        yield Footer()

    # Handle typing: Input has changed so update predictions
    async def on_input_changed(self, event: Input.Changed) -> None:
        """ Re- run suggestion logic everytime user types."""
        fragment = event.value
        self.user_text = fragment

        if not fragment.strip():
            self.suggestions = []
            return

        start = time.perf_counter()
        preds = self.hp.suggest(fragment)
        self.latency = time.perf_counter() - start
        self.suggestions = preds

    # Reactive state (watcher functions) ---------------------------------------
    def watch_suggestions(self, suggestions):
        """Refresh the right-hand suggestion widget."""
        self.query_one(SuggestionPanel).update_predictions(suggestions)

    def watch_latency(self, latency):
        """Update the latency indicator when the number changes."""
        self.query_one(TypingLatency).set_latency(latency)

    # Actions ----------------------------------------------------------------------
    # triggered by keyboard shortcuts
    def action_accept_top(self):
        """TAB = accept the top suggestion."""
        if not self.suggestions:
            return
        top_items, _ = self.suggestions[0]
        self.accept_word(top_items)

    def action_save_session(self):
        """Ctrl+S = manually save user text + learned model."""
        self.save_state()
        self.query_one("#status").update("[green]Saved[/green]")

    def action_reset_context(self):
        """Ctrl+R = reset local learner memory."""
        self.context.reset()
        self.query_one(ContextView).update_context(self.context)
        self.query_one("#status").update("[yellow]Context cleared[/yellow]")

    # Accept Suggestions (using 1-5 keys) -----------------------------------------------------
    async def on_key(self, event: events.Key) -> None:
        if event.key.isdigit():
            idx = int(event.key) - 1
            if 0 <= idx < len(self.suggestions):
                word, _ = self.suggestions[idx]
                self.accept_word(word)

    # Word acceptance logic ---------------------------------------------------------------
    def accept_word(self, word):
        """
        When user accepts a suggestion:
         - train the predictor
         - update personal context
         - append to input field
        """
        self.hp.accept(word)
        self.context.learn(word)
        Log.write(f"accepted: {word}")

        # update text field
        input_widget = self.query_one(Input)
        new_text = (input_widget.value + " " + word).strip()
        input_widget.value = new_text
        self.user_text = new_text

        # refresh learned context UI panel
        self.query_one(ContextView).update_context(self.context)

    # Persistence (saving + loading) --------------------------------------------------------
    def save_state(self):
        """Save current text and model to disk at path initialised above."""
        SESSION_PATH.parent.mkdir(exist_ok=True)
        with open(SESSION_PATH, "w", encoding="utf8") as f:
            f.write(self.user_text)
        with open(MODEL_PATH, "wb") as f:
            import pickle
            pickle.dump(self.hp, f)

    def load_state(self):
        """Load previously saved text and model if available, using path initialised above."""
        if MODEL_PATH.exists():
            import pickle
            self.hp = pickle.load(open(MODEL_PATH, "rb"))
        if SESSION_PATH.exists():
            self.user_text = SESSION_PATH.read_text(encoding="utf8")

    # Startup hook ---------------------------------------------------------------------
    def on_mount(self):
      """Once UI is ready update context panel with loaded state."""
        self.query_one(ContextView).update_context(self.context)


if __name__ == "__main__":
    TUIAutocompleter().run()

