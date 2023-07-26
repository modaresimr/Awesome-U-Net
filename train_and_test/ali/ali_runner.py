import os
import comet_ml
import sys
sys.path.append('../..')
if 1:
    from utils import (
        show_sbs,
        load_config,
        _print,
    )


#!/usr/bin/env python3


def ali_runner(CONFIG_FILE_PATH):
    config = load_config(CONFIG_FILE_PATH)

    config['training']['epochs'] = 10

    import ali_common
    ali_common.execute(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 ali_runner.py <CONFIG_FILE_PATH>")
        sys.exit(1)

    ali_runner(sys.argv[1])
