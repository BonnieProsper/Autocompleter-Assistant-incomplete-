# intelligent_autocompleter/plugins/plugin_loader.py
# plugin loader that loads .py files from a directory and instantiates plugin classes.
# intentionally simple/explicit for now
"""
PluginLoader - scans a path and registers plugin instances into a PluginRegistry.
Usage:
  reg = PluginRegistry()
  loader = PluginLoader(path="intelligent_autocompleter/plugins", registry=reg, cfg_map=my_cfg)
  loader.load_all()
  reg.call_init()
"""

import os
import importlib.util
import sys
import traceback 
from typing import Optional, Dict, List
from .registry import PluginRegistry
from .base import PluginBase

class PluginLoader:
    def __init__(self, path: str, registry: Optional[PluginRegistry] = None, cfg_map: Optional[Dict[str, dict]] = None):
        """
        path: directory containing plugins (modules)
        cfg_map: dict(plugin_name to config) to pass into plugin constructors
        """
        self.path = os.path.abspath(path)
        self.registry = registry or PluginRegistry()
        self.cfg_map = cfg_map or {}

    def discover(self) -> List[str]:
        """Return list of candidate modules (module name, path to file)."""
        out = []
        if not os.path.isdir(self.path):
            return out
        for nm in sorted(os.listdir(self.path)):
            if nm.startswith("_"):
                continue
            py = os.path.join(self.path, nm + ".py")
            pkg_init = os.path.join(self.path, nm, "__init__.py")
            if os.path.exists(py):
                out.append((nm, py))
            elif os.path.exists(pkg_init):
                out.append((nm, pkg_init))
        return out

    def load_module(self, mod_name: str, path: str):
        spec = importlib.util.spec_from_file_location(f"intelligent_autocompleter.plugins.{mod_name}", path)
        if spec is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)  # type: ignore
            return mod
        except Exception:
            print(f"[plugin-loader] failed to import {mod_name}")
            traceback.print_exc()
            return None

    def load_all(self, verbose: bool = False):
        for nm, p in self.discover():
            mod = self.load_module(nm, p)
            if not mod:
                continue
            # try register() first
            if hasattr(mod, "register") and callable(getattr(mod, "register")):
                try:
                    mod.register(self.registry)
                    if verbose:
                        print(f"[plugin-loader] registered {nm} via register()")
                    continue
                except Exception:
                    traceback.print_exc()
            # otherwise try to find a Plugin/PluginClass
            for objname in dir(mod):
                obj = getattr(mod, objname)
                try:
                    if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                        cfg = self.cfg_map.get(getattr(obj, "name", objname), None)
                        inst = obj(cfg)
                        self.registry.register(inst)
                        if verbose:
                            print(f"[plugin-loader] instantiated plugin {objname} from {nm}")
                except Exception:
                    traceback.print_exc()

    def load_one(self, module_name: str, verbose: bool = False):
        """Load a single plugin by module name (filename without .py or package dir)."""
        for nm, p in self.discover():
            if nm == module_name:
                mod = self.load_module(nm, p)
                if mod:
                    if hasattr(mod, "register") and callable(getattr(mod, "register")):
                        mod.register(self.registry)
                        if verbose:
                            print(f"[plugin-loader] registered {nm} via register()")
                        return True
                    for objname in dir(mod):
                        obj = getattr(mod, objname)
                        if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                            cfg = self.cfg_map.get(getattr(obj, "name", objname), None)
                            inst = obj(cfg)
                            self.registry.register(inst)
                            if verbose:
                                print(f"[plugin-loader] loaded plugin {objname}")
                            return True
        raise FileNotFoundError(module_name)
