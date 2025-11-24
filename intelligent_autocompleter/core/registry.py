"""
registry.py
Central extensible registry for commands, autocomplete providers,
validators, and reasoning plugins.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Any
from intelligent_autocompleter.core.reasoner import ReasoningResult

class CommandRegistry:
    """
    Unified plugin/command registry for the Autocompleter Assistant.
    Supports:
     - command handlers
     - autocomplete providers
     - input validators
     - reasoning plugins (returning ReasoningResult objects)
    The registry ensures consistent behavior and isolates failures
    in plugin-provided logic.
    """
    def __init__(self):
        # command_name -> handler function
        self.commands: Dict[str, Callable[..., Any]] = {}

        # command_name -> list of autocomplete providers
        self.autocompletes: Dict[str, List[Callable[[str], List[str]]]] = {}

        # command_name -> validator function
        self.validators: Dict[str, Callable[[str], Optional[str]]] = {}

        # reasoning plugins -> callable(str) -> ReasoningResult | List[ReasoningResult] | None
        self.reasoning_plugins: List[Callable[[str], Any]] = []


    # COMMANDS ----------------------------------------------------------------
    def add_command(self, name: str, handler: Callable[..., Any]) -> None:
        """
        Registers a named command with its handler.
        Handler signature: handler(args...) -> Any
        """
        self.commands[name] = handler

    # AUTOCOMPLETE PROVIDERS ------------------------------------------------------
    def add_autocomplete(self, command_name: str, provider: Callable[[str], List[str]]) -> None:
        """
        Registers an autocomplete provider for a command.
        Multiple providers can exist for the same command.
        """
        self.autocompletes.setdefault(command_name, []).append(provider)

    def get_autocomplete_suggestions(self, command_name: str, text: str) -> List[str]:
        """
        Runs all autocomplete providers for a command.
        """
        providers = self.autocompletes.get(command_name, [])
        results: List[str] = []
        for provider in providers:
            try:
                out = provider(text)
                if out:
                    results.extend(out)
            except Exception:
                continue
        return results

    # VALIDATORS -----------------------------------------------------------------------
    def add_validator(self, command_name: str, validator: Callable[[str], Optional[str]]) -> None:
        """
        Adds a validator for a command.
        Validator should return:
        - None if valid
        - error message string otherwise
        """
        self.validators[command_name] = validator

    def validate(self, command_name: str, text: str) -> Optional[str]:
        """
        Runs a validator if it exists for the given command.
        """
        validator = self.validators.get(command_name)
        if not validator:
            return None
        try:
            return validator(text)
        except Exception:
            return "Unknown validation error."

    # REASONING PLUGINS -----------------------------------------------------
    def add_reasoning_plugin(self, plugin: Callable[[str], Any]) -> None:
        """
        Adds a reasoning plugin.
        Plugin should return:
         - ReasoningResult
         - list of ReasoningResult
         - None
        """
        self.reasoning_plugins.append(plugin)

    def run_reasoning(self, input_text: str) -> List[ReasoningResult]:
        """
        Runs all reasoning plugins and returns results.
        """
        results: List[ReasoningResult] = []

        for plugin in self.reasoning_plugins:
            try:
                out = plugin(input_text)

                if isinstance(out, ReasoningResult):
                    results.append(out)
                elif isinstance(out, list):
                    # filter only valid ReasoningResult items
                    for item in out:
                        if isinstance(item, ReasoningResult):
                            results.append(item)
            except Exception:
                # Plugin failures must never crash the engine
                continue
        return results

    # INTROSPECTION UTILITIES -------------------------------------------------
    def list_commands(self) -> List[str]:
        """Return a sorted list of registered command names."""
        return sorted(self.commands.keys())

    def list_plugins(self) -> List[str]:
        """Return a list of registered plugin callables for debugging."""
        return [p.__name__ for p in self.reasoning_plugins]

