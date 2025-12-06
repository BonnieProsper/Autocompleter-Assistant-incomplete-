# intelligent_autocompleter/plugins/code_assist_plugin.py
from typing import List, Dict, Any, Tuple
from ..base import PluginBase

# Candidate type: (suggested_text, score)
Candidate = Tuple[str, float]


class CodeAssist(PluginBase):
    """
    CodeAssist plugin suggests small common code snippets.
    """

    name = "code_assist"
    version = "0.2"

    SNIPPETS: Dict[str, str] = {
        "for": "for i in range(n):\n    pass",
        "if": "if cond:\n    pass",
        "def": "def function_name(args):\n    '''docstring'''\n    return None",
        "import": "import os\nimport sys",
        "try": "try:\n    pass\nexcept Exception as e:\n    print(e)",
    }

    def on_suggest(
        self, fragment: str, candidates: List[Candidate], bundle: Dict[str, Any]
    ) -> List[Candidate]:
        """
        Suggest a snippet (word,score) list. Plugins should return list[(str,float)].
        """
        tok = fragment.strip().split()[-1] if fragment.strip() else ""
        out: List[Candidate] = list(candidates)
        if tok in self.SNIPPETS:
            out.insert(0, (self.SNIPPETS[tok], 0.9))
        return out


# helper registration function: optional convenience for plugin loader
def register(registry):
    """
    Called by plugin loader if module defines register(registry).
    """
    registry.register(CodeAssist())
