MODELS_REGISTRY = {}

def register_models(name):
    def wrapper(func):
        MODELS_REGISTRY[name] = func
        return func
    return wrapper