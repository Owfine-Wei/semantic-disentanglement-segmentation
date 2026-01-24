# models/__init__.py
import os
import importlib
from .registry import CONFIGS_REGISTRY # 从中转站拿字典

# 自动扫描逻辑保持不变
config_dir = os.path.dirname(__file__)
for file in os.listdir(config_dir):
    if file.endswith(".py") and file not in ["__init__.py", "registry.py"]:
        module_name = f"configs.{file[:-3]}"
        importlib.import_module(module_name)

def get_config(name):
    if name not in CONFIGS_REGISTRY:
        raise KeyError(f"模型 '{name}' 尚未注册！可用: {list(CONFIGS_REGISTRY.keys())}")
    return CONFIGS_REGISTRY[name]