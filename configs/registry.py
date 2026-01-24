# configs/registry.py

CONFIGS_REGISTRY = {}

def register_configs(name):
    def wrapper(cls):
        CONFIGS_REGISTRY[name] = cls
        return cls
    return wrapper