# intelligent_autocompleter/plugins/loader.py
# plugin loader that loads .py files from a directory and instantiates plugin classes.
# intentionally simple/explicit for now

import os
import importlib.util
import sys
from typing import Optional, List
from .registry import PluginRegistry
from .base import PluginBase

class PluginLoader:
    def __init__(self, path: str, registry: Optional[PluginRegistry] = None, cfg_map: dict = None):
        """
        path: directory containing plugins (modules)
        cfg_map: dict(plugin_name to config) to pass into plugin constructors
        """
        self.path = os.path.abspath(path)
        self.registry = registry or PluginRegistry()
        self.cfg_map = cfg_map or {}

    def discover(self) -> List[str]:
        if not os.path.isdir(self.path):
            return []
        names = []
        for nm in os.listdir(self.path):
            if nm.startswith("_"):
                continue
            if nm.endswith(".py"):
                names.append(nm[:-3])
            elif os.path.isdir(os.path.join(self.path, nm)) and os.path.exists(os.path.join(self.path, nm, "__init__.py")):
                names.append(nm)
        return names

    def load_module(self, filepath: str, modname: str):
        spec = importlib.util.spec_from_file_location(modname, filepath)
        if spec is None:
            return None
        m = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(m)  # type ignore
            return m
        except Exception:
            # fail noisily but continue
            print(f"[plugin-loader] failed to import {modname} ({filepath})")
            return None

    def load_all(self):
        names = self.discover()
        for nm in names:
            p = os.path.join(self.path, nm + ".py")
            pkg = os.path.join(self.path, nm, "__init__.py")
            target = p if os.path.exists(p) else pkg
            if not os.path.exists(target):
                continue
            mod = self.load_module(target, f"plugin_{nm}")
            if not mod:
                continue
            # instantiate classes that subclass PluginBase
            for objname in dir(mod):
                obj = getattr(mod, objname)
                try:
                    if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                        cfg = self.cfg_map.get(getattr(obj, "name", objname), None)
                        inst = obj(cfg)
                        self.registry.register(inst)
                except Exception:
                    # weird plugin so ignore
                    continue

    def load_one(self, module_name: str):
        # load a single module by file name or folder name for convenience
        target_py = os.path.join(self.path, module_name + ".py")
        target_pkg = os.path.join(self.path, module_name, "__init__.py")
        target = target_py if os.path.exists(target_py) else target_pkg
        if not os.path.exists(target):
            raise FileNotFoundError(target)
        mod = self.load_module(target, f"plugin_{module_name}")
        if mod is None:
            raise ImportError(module_name)
        for objname in dir(mod):
            obj = getattr(mod, objname)
            if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                cfg = self.cfg_map.get(getattr(obj, "name", objname), None)
                inst = obj(cfg)
                self.registry.register(inst)
        return True
