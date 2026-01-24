# configs/__init__.py
import os
import importlib
from .registry import CONFIGS_REGISTRY 

config_dir = os.path.dirname(__file__)
for file in os.listdir(config_dir):
    if file.endswith(".py") and file not in ["__init__.py", "registry.py"]:
        module_name = f"configs.{file[:-3]}"
        importlib.import_module(module_name)

def get_config(name):
    if name not in CONFIGS_REGISTRY:
        raise KeyError(f"Config '{name}' is not registered")
    return CONFIGS_REGISTRY[name]