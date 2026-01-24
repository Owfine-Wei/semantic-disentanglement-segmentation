# models/__init__.py
import os
import importlib

MODELS_CONFIG_REGISTRY = {}

def register_models(name):
    def wrapper(function):
        if name in MODELS_CONFIG_REGISTRY:
            print(f"Warning: {name} is already registered. Overwriting...")
        MODELS_CONFIG_REGISTRY[name] = function
        return function
    return wrapper

def get_model(name):
    return 

