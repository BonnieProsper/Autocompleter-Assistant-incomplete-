# intelligent_autocompleter/plugins/emoji_plugin.py
from .base import PluginBase, Candidate
from typing import List, Dict, Any

class EmojiPlugin(PluginBase):
    name = "emoji"
    version = "0.1"

    EMOJI_MAP = {
        "smile": "😊",
        "happy": "😊",
        "sad": "😢",
        "rocket": "🚀",
        "fire": "🔥",
        "ok": "👌",
        "love": "❤️",
    }

    def on_suggest(self, fragment: str, candidates: List[Candidate], bundle: Dict[str, Any]) -> List[Candidate]:
        frag = fragment.lower().strip()
        out = list(candidates)
        for k, e in self.EMOJI_MAP.items():
            if frag.endswith(k) or k in frag:
                out.append((e, 0.45))
        return out

def register(reg):
    reg.register(EmojiPlugin())

