# plugins/example_plugin.py

def register(registry):
    """
    Plugins define one public function: register(registry)
    Can attach any wanted functionality here.
    """

    # Register a new command
    def hello(args):
        print(f"Hello from plugin! Args: {args}")

    registry.register_command("hello", hello)

    # Register autocomplete
    registry.register_autocomplete("hello", lambda _: ["world", "there", "friend"])

    # Register validator
    registry.register_validator("hello", lambda text: None)

    # Register reasoning hook
    def reasoning(text):
        if text.startswith("hello "):
            return "Plugin says: try 'hello world'!"
        return None

    registry.register_reasoning_hook(reasoning)
