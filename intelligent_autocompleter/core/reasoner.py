import re
from typing import List
from cli.ui.colors import yellow, red, green

from intelligent_autocompleter.core.semantic_engine import SemanticEngine

class CommandReasoner:
    """
    A reasoning engine that analyzes user input to provide contextual suggestions 
    based on predefined rules and external reasoning hooks.
    """
    def __init__(self, registry):
        self.registry = registry  # Registry that holds additional reasoning hooks
        self.semantic = SemanticEngine()

    def analyze_input(self, user_input: str) -> List[str]:
        """
        Analyzes the given user input and generates relevant suggestions or warnings.
        """
        suggestions = []
      
        # Run built-in reasoning checks
        suggestions.extend(self.flag_dangerous(user_input))
        suggestions.extend(self.suggest_autocorrections(user_input))
        suggestions.extend(self.predict_next_argument(user_input))
        
        # Run plugin-specific reasoning hooks
        suggestions.extend(self.registry.run_reasoning(user_input))

        # Semantic intelligence layer
        semantic_cmd = self.semantic_engine.predict_command(user_input)
        if semantic_cmd:
            suggestions.append(green(f"AI Suggestion → {semantic_cmd}"))

        return suggestions

    def flag_dangerous(self, user_input: str) -> List[str]:
        """
        Flags potentially dangerous commands that could result in data loss or system issues.
        """
        dangerous_commands = ["rm -rf", "sudo rm", "mkfs", "shutdown"]
        warnings = []
        for dangerous_command in dangerous_commands:
            if dangerous_command in user_input:
                warnings.append(red(f"⚠ WARNING: '{user_input}' seems risky."))
        return warnings

    def suggest_autocorrections(self, user_input: str) -> List[str]:
        """
        Suggests possible corrections for common typos or misused commands.
        """
        corrections = []
        if "gti " in user_input:
            corrections.append(yellow("Did you mean 'git'?"))
        if re.match(r"pip instal ", user_input):
            corrections.append(yellow("Did you mean 'pip install'?"))

        return corrections

    def predict_next_argument(self, user_input: str) -> List[str]:
        """
        Suggests likely next commands based on the user's current input, to assist with 
        completing commonly used command sequences.
        """
        predictions = []
        if user_input.startswith("git "):
            common_git_commands = ["commit", "push", "pull", "status", "branch"]
            predictions.append(green("Next likely git commands: " + ", ".join(common_git_commands)))

        return predictions
