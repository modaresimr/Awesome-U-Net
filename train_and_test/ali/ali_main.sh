#!/bin/bash

for f in $(ls ali_configs_new/isic2018);do
	if [[ $f != '_*' ]]; then 
		echo python3 ali_runner.py ./ali_configs_new/isic2018/$f
	fi
done
#python3 ali_runner.py ./ali_configs_new/isic2018/unet_5.yaml
# python3 ali_runner.py ./ali_configs_new/isic2018/unet_7.yaml

python3 ali_runner.py ./ali_configs_new/isic2018/adapt_unet_3-9.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/unet_3.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/unet_11.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/unet_13.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/unet_15.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/adapt_unet_5-5.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/adapt_unet_7-7.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/adapt_unet_3-3.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/acda_unet_3-3.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/acda_unet_3-9.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/acda_unet_5-5.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/acda_unet_7-7.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/acda_unet_9-9.yaml

# python3 ali_runner.py ./ali_configs_new/isic2018/adapt_unet_9-9.yaml

python3 ali_runner.py ./ali_configs_new/isic2018/missformer.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/multiresunet.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/transunet.yaml


python3 ali_runner.py ./ali_configs_new/isic2018/unet.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/acda_uctransnet_7.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/adapt_uctransnet_7.yaml
python3 ali_runner.py ./ali_configs_new/isic2018/uctransnet.yaml

# python3 ali_runner.py ./ali_configs_new/isic2018/attunet.yaml
# python3 ali_runner.py ./ali_configs_new/isic2018/resunet.yaml

# python3 ali_runner.py ./ali_configs_new/isic2018/unetpp.yaml


# for f in $(ls ali_configs_new/isic2018');do
# 	python3 ali_runner.py ./ali_configs/isic2018/$f
# done

# # for ((i=3; i<=13; i+=2))
# # do
# #     #python3 ali_runner.py ./ali_configs/segpc/segpc2021_uctransnet_$i.yaml
# #     python3 ali_runner.py ./ali_configs/isic2018/isic2018_uctransnet_$i.yaml
# #     # exit 1
# # done


# # python3 ali_runner.py ./ali_configs/segpc/segpc2021_unet.yaml
# # python3 ali_runner.py ./ali_configs/segpc/segpc2021_unetpp.yaml
# # python3 ali_runner.py ./ali_configs/segpc/segpc2021_uctransnet.yaml


# # python3 ali_runner.py ./ali_configs/segpc/segpc2021_transunet.yaml
# # python3 ali_runner.py ./ali_configs/segpc/segpc2021_resunet.yaml
# # python3 ali_runner.py ./ali_configs/segpc/segpc2021_multiresunet.yaml
# # python3 ali_runner.py ./ali_configs/segpc/segpc2021_missformer.yaml
# # python3 ali_runner.py ./ali_configs/segpc/segpc2021_attunet.yaml


# python3 ali_runner.py ali_configs/segpc_2class/segpc2021_2c_adaptuctransnet_7.yaml
# python3 ali_runner.py ali_configs/segpc_2class/segpc2021_2c_unet.yaml
# python3 ali_runner.py ali_configs/segpc_2class/segpc2021_2c_uctransnet.yaml
# python3 ali_runner.py ali_configs/segpc_2class/segpc2021_2c_attunet.yaml
# python3 ali_runner.py ali_configs/segpc_2class/segpc2021_2c_resunet.yaml
# python3 ali_runner.py ali_configs/segpc_2class/segpc2021_2c_unetpp.yaml
# python3 ali_runner.py ali_configs/segpc_2class/segpc2021_2c_missformer.yaml
# python3 ali_runner.py ali_configs/segpc_2class/segpc2021_2c_transunet.yaml
# python3 ali_runner.py ali_configs/segpc_2class/segpc2021_2c_multiresunet.yaml







# python3 ali_runner.py ali_configs/segpc/segpc2021_adaptuctransnet_7.yaml
# python3 ali_runner.py ./ali_configs/segpc/segpc2021_unet.yaml
# python3 ali_runner.py ./ali_configs/segpc/segpc2021_uctransnet.yaml
# python3 ali_runner.py ./ali_configs/segpc/segpc2021_unetpp.yaml
# python3 ali_runner.py ./ali_configs/segpc/segpc2021_transunet.yaml
# python3 ali_runner.py ./ali_configs/segpc/segpc2021_resunet.yaml
# python3 ali_runner.py ./ali_configs/segpc/segpc2021_multiresunet.yaml
# python3 ali_runner.py ./ali_configs/segpc/segpc2021_missformer.yaml
# python3 ali_runner.py ./ali_configs/segpc/segpc2021_attunet.yaml



