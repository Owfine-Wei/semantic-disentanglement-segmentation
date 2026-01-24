# models/registry.py

# 真正的登记中心
MODELS_REGISTRY = {}

def register_models(name):
    def wrapper(func):
        MODELS_REGISTRY[name] = func
        return func
    return wrapper