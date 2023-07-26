import sys
sys.path.append('../..')
from utils import (
    show_sbs,
    load_config,
    _print,
)
import os

import comet_ml


#!/usr/bin/env python3


def ali_runner(i):
    CONFIG_FILE_PATH = f"./ali_configs/isic2018_uctransnet_{i}.yaml"
    config = load_config(CONFIG_FILE_PATH)

    config['training']['epochs'] = 50
    net_config = config['model']['params']
    # net_config['num_bases'] = 6
    # net_config['DCFD_kernel_size'] = i

    import ali_common
    ali_common.execute(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 ali_runner.py <i>")
        sys.exit(1)

    i = int(sys.argv[1])
    ali_runner(i)
