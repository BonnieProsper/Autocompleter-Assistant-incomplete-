# utils/registry.py

from typing import Callable, List, Dict, Optional

class CommandRegistry:
    """
    Central registry for commands, autocomplete providers,
    validators, and reasoning hooks.
    """
    def __init__(self):
        self.commands: Dict[str, Callable] = {}
        self.autocomplete: Dict[str, Callable[[str], List[str]]] = {}
        self.validators: Dict[str, Callable[[str], Optional[str]]] = {}
        self.reasoning_hooks: List[Callable[[str], Optional[str]]] = []

    def register_command(self, name: str, handler: Callable):
        self.commands[name] = handler

    def register_autocomplete(self, name: str, provider: Callable):
        self.autocomplete[name] = provider

    def register_validator(self, name: str, validator: Callable):
        self.validators[name] = validator

    def register_reasoning_hook(self, hook: Callable):
        self.reasoning_hooks.append(hook)

    def run_reasoning(self, text: str) -> List[str]:
        """
        Run all reasoning hooks and aggregate their suggestions.
        """
        output = []
        for hook in self.reasoning_hooks:
            try:
                suggestion = hook(text)
                if suggestion:
                    output.append(suggestion)
            except Exception:
                continue
        return output

