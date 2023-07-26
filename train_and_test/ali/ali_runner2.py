import comet_ml
import os

import sys
sys.path.append('../..')
from utils import (
    show_sbs,
    load_config,
    _print,
)


i = 9
#!/usr/bin/env python3
CONFIG_FILE_PATH = f"./ali_configs/isic2018_uctransnet_{i}.yaml"

config = load_config(CONFIG_FILE_PATH)


def class_by_name(clazz):
    module_name, class_name = clazz.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


print(class_by_name(config['model']['class']))
