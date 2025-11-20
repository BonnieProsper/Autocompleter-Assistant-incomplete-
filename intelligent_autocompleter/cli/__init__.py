from intelligent_autocompleter.plugins import PluginLoader, PluginRegistry
plugins_dir = os.path.join(os.path.dirname(__file__), "plugins", "examples")
self.plugin_registry = PluginRegistry()
self.plugin_loader = PluginLoader(plugins_dir, registry=self.plugin_registry)
self.plugin_loader.load_all()

