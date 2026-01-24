# models/__init__.py
import os
import importlib
from .registry import MODELS_REGISTRY # 从中转站拿字典

# 自动扫描逻辑保持不变
model_dir = os.path.dirname(__file__)
for file in os.listdir(model_dir):
    if file.endswith(".py") and file not in ["__init__.py", "registry.py"]:
        module_name = f"models.{file[:-3]}"
        importlib.import_module(module_name)

def get_model(name):
    if name not in MODELS_REGISTRY:
        raise KeyError(f"模型 '{name}' 尚未注册！可用: {list(MODELS_REGISTRY.keys())}")
    return MODELS_REGISTRY[name]