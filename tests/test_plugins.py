# tests/test_plugins.py
from pathlib import Path
from intelligent_autocompleter.plugins.registry import PluginRegistry
from intelligent_autocompleter.plugins.loader import PluginLoader


def test_load_plugins():
    pdir = Path(__file__).parents[1] / "intelligent_autocompleter" / "plugins"
    reg = PluginRegistry()
    loader = PluginLoader(str(pdir), registry=reg)
    loader.load_all(verbose=False)
    reg.call_init()
    # plugin names present (one or both)
    names = reg.names()
    assert isinstance(names, list)
    # run pipeline
    out = reg.run_suggest_pipeline("I am happy", [], {})
    assert isinstance(out, list)
