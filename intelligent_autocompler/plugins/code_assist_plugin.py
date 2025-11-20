# intelligent_autocompleter/plugins/code_assist_plugin.py
from ..base import PluginBase

class CodeAssist(PluginBase):
    """
    CodeAssist plugin has a predefined set of code pieces to assist the user
    with faster coding for common structures like loops, conditions,
    function definitions, and imports.
    """
    # plugin version/name
    name = "code_assist"
    version = "0.1"

    # predefined common code structure set
    SNIPPETS = {
        "for": "for i in range(n):\n    pass", # for loop template
        "if": "if cond:\n    pass",            # if statement template
        "def": "def function_name(args):\n    '''docstring'''\n    return None", # function definition + docstring
        "import": "import os\nimport sys",     # common imports (system and os libraries)
    }

    def on_suggest(self, bundle, suggestions):
        """
        Suggests a code snippet based on the last token typed by the user.
        Parameters:
        - bundle (dict): Contains various contextual information like the last token typed.
        - suggestions (list): Existing suggestions that the plugin can enhance.
        Returns:
        - list: Updated list of suggestions with an additional code snippet if applicable.
        """
        last = bundle.get("last_token", "")
        if not last:
            return suggestions
        key = last.strip()
        if key in self.SNIPPETS:
            suggestions = suggestions.copy()
            suggestions.insert(0, (self.SNIPPETS[key], 0.9))
        return suggestions

