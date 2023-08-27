import sys
import yaml
sys.path.append('../..')
from utils import (
    show_sbs,
    load_config,
    _print,
)


def class_by_name(clazz):
    module_name, class_name = clazz.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def save_config(config, config_filepath):
    try:
        with open(config_filepath, 'w') as file:
            yaml.dump(config, file)
    except FileNotFoundError:
        _print(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)
