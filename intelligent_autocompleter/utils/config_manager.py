# config_manager.py - JSON config manager

import json
import os


class Config:
    def __init__(self, path="config.json"):
        self.path = path
        self.data = {
            "balance": 0.6,  # markov/embedding weighting
            "autosave": True,
            "max_suggestions": 5,
            "theme": "default",
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf8") as f:
                    self.data.update(json.load(f))
            except Exception:
                pass
        else:
            self.save()

    def save(self):
        with open(self.path, "w", encoding="utf8") as f:
            json.dump(self.data, f, indent=2)

    def show(self):
        for k, v in self.data.items():
            print(f"{k:15} = {v}")

    def set(self, key, val):
        if key not in self.data:
            print("No such option")
            return
        self.data[key] = type(self.data[key])(val)
        self.save()
