
# 1. 建立一个全局字典，用来存：{ "数据集名称": 配置类 }
DATASET_CONFIG_REGISTRY = {}

# 2. 定义装饰器：它的唯一作用就是在程序启动时，把类塞进上面的字典
def register_config(name):
    def wrapper(cfg_class):
        if name in DATASET_CONFIG_REGISTRY:
            print(f"Warning: {name} is already registered. Overwriting...")
        DATASET_CONFIG_REGISTRY[name] = cfg_class
        return cfg_class
    return wrapper

def get_config(name):
    """安全地获取配置类"""
    if name not in DATASET_CONFIG_REGISTRY:
        raise KeyError(f"配置 '{name}' 尚未注册！请检查 configs/ 文件夹下的文件名和装饰器。")
    return DATASET_CONFIG_REGISTRY[name]