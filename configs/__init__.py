# configs/__init__.py
import os
import importlib

# 自动扫描当前 configs 文件夹
config_dir = os.path.dirname(__file__)

for file in os.listdir(config_dir):
    # 只要是 .py 文件且不是 __init__.py 本身
    if file.endswith(".py") and file != "__init__.py":
        module_name = f"configs.{file[:-3]}"
        # 动态导入。一旦导入，文件里的 @register_config 就会生效
        importlib.import_module(module_name)