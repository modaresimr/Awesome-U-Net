#!/bin/bash

# for ((i=3; i<=13; i+=2))
# do
#     #python3 ali_runner.py ./ali_configs/segpc/segpc2021_uctransnet_$i.yaml
#     python3 ali_runner.py ./ali_configs/isic2018/isic2018_uctransnet_$i.yaml
#     # exit 1
# done


# python3 ali_runner.py ./ali_configs/segpc/segpc2021_unet.yaml
# python3 ali_runner.py ./ali_configs/segpc/segpc2021_unetpp.yaml
# python3 ali_runner.py ./ali_configs/segpc/segpc2021_uctransnet.yaml


python3 ali_runner.py ./ali_configs/segpc/segpc2021_transunet.yaml
python3 ali_runner.py ./ali_configs/segpc/segpc2021_resunet.yaml
python3 ali_runner.py ./ali_configs/segpc/segpc2021_multiresunet.yaml
python3 ali_runner.py ./ali_configs/segpc/segpc2021_missformer.yaml
python3 ali_runner.py ./ali_configs/segpc/segpc2021_attunet.yaml
