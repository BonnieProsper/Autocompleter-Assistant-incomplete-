# intelligent_autocompleter/plugins/code_assist_plugin.py
from ..base import PluginBase
from typing import List, Dict, Any

Candidate = Tuple[str, float]

class CodeAssist(PluginBase):
    """
    CodeAssist plugin has a predefined set of code pieces to assist the user
    with faster coding for common structures like loops, conditions,
    function definitions, and imports.
    """
    # plugin version/name
    name = "code_assist"
    version = "0.2"

    # predefined common code structure set
    SNIPPETS = {
        "for": "for i in range(n):\n    pass", # for loop template
        "if": "if cond:\n    pass",            # if statement template
        "def": "def function_name(args):\n    '''docstring'''\n    return None", # function definition + docstring
        "import": "import os\nimport sys",     # common imports (system and os libraries)
         "try": "try:\n    pass\nexcept Exception as e:\n    print(e)", # try and exception error template
    }

    def on_suggest(self, fragment: str, candidates: List[Candidate], bundle: Dict[str, Any]) -> List[Candidate]:
        """
        Suggests a code snippet based on the last token typed by the user.
        """
        tok = fragment.strip().split()[-1] if fragment.strip() else ""
        out = list(candidates)
        if tok in self.SNIPPETS:
            out.insert(0, (self.SNIPPETS[tok], 0.9))
        return out

    def register(reg):
        """Correct plugin loader entrypoint."""
        reg.register(CodeAssist())
