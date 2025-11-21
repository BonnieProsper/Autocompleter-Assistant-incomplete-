# intelligent_autocompleter/plugins/__init__.py
"""
Plugin package for intelligent_autocompleter.
Expose loader + registry + base plugin so callers can import conveniently.
"""

from .loader import PluginLoader
from .registry import PluginRegistry
from .base import PluginBase

__all__ = ["PluginLoader", "PluginRegistry", "PluginBase"]
