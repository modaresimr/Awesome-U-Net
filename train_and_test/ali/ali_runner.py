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
    parent = os.path.dirname(CONFIG_FILE_PATH)
    config = load_config(parent + "/_dataset.yaml")
    config = {**config, **load_config(parent + "/_common.yaml")}
    config = {**config, **load_config(CONFIG_FILE_PATH)}
    config['config'] = {'name': parent.split("/")[-1] + "_" + CONFIG_FILE_PATH.split("/")[-1].split(".")[0]}
    config['model']['save_dir'] = '../../saved_models/' + config['config']['name']
    try:
        os.removedirs(config['model']['save_dir'])
    except:
        pass
    config['training']['epochs'] = 100

    import ali_common
    ali_common.execute(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 ali_runner.py <CONFIG_FILE_PATH>")
        sys.exit(1)

    ali_runner(sys.argv[1])
