# main.py - to remove/change

import os
import importlib.util
from cli.core.registry import CommandRegistry
from cli.core.reasoner import CommandReasoner
from cli.ui.prompts import run_cli_loop

def load_plugins(registry: CommandRegistry):
    plugins_dir = os.path.join(os.path.dirname(__file__), "plugins")

    for fname in os.listdir(plugins_dir):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue

        path = os.path.join(plugins_dir, fname)
        spec = importlib.util.spec_from_file_location(fname[:-3], path)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            if hasattr(module, "register"):
                module.register(registry)
                print(f"Loaded plugin: {fname}")
        except Exception as e:
            print(f"Failed to load plugin {fname}: {e}")

def main():
    registry = CommandRegistry()
    load_plugins(registry)
    reasoner = CommandReasoner(registry)

    run_cli_loop(registry, reasoner)

if __name__ == "__main__":
    main()
