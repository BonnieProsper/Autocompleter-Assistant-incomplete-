# utils/registry.py

from typing import Callable, List, Dict, Optional

class CommandRegistry:
    """
    A central manager for commands, autocomplete providers, input validators, 
    and reasoning hooks. This class allows registering and managing various 
    functionalities related to command processing.
    """
    def __init__(self):
        self.commands: Dict[str, Callable] = {}  # Stores command names and their handlers
        self.autocompletes: Dict[str, Callable[[str], List[str]]] = {}  # Stores autocomplete providers for commands
        self.validators: Dict[str, Callable[[str], Optional[str]]] = {}  # Stores input validators for commands
        self.reasoning_hooks: List[Callable[[str], Optional[str]]] = []  # Stores reasoning hooks for command suggestions

    def add_command(self, command_name: str, handler: Callable):
        """
        Registers a new command with its handler function.
        """
        self.commands[command_name] = handler

    def add_autocomplete_provider(self, command_name: str, provider: Callable):
        """
        Registers an autocomplete provider for a specific command.
        """
        self.autocompletes[command_name] = provider

    def add_input_validator(self, command_name: str, validator: Callable):
        """
        Registers an input validator for a specific command.
        """
        self.validators[command_name] = validator

    def add_reasoning_hook(self, hook: Callable):
        """
        Registers a new reasoning hook to generate suggestions based on the command input.
        """
        self.reasoning_hooks.append(hook)

    def execute_reasoning_hooks(self, input_text: str) -> List[str]:
        """
        Executes all registered reasoning hooks and aggregates their suggestions.
        """
        suggestions = []
        for hook in self.reasoning_hooks:
            try:
                suggestion = hook(input_text)
                if suggestion:
                    suggestions.append(suggestion)
            except Exception:
                continue
        return suggestions
