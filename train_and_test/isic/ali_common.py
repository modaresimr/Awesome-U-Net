from __future__ import print_function, division
import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from comet_ml import ConfusionMatrix, Experiment, init
from comet_ml.integration.pytorch import log_model
import datetime
import random
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.optim import Adam, SGD
from losses import DiceLoss, DiceLossWithLogtis
from torch.nn import BCELoss, CrossEntropyLoss

import ali_utils


def execute(config):
    key = config['run']['key']
    model_class = ali_utils.class_by_name(config['model']['class'])
    dataset_class = ali_utils.class_by_name(config['dataset']['class'])

    # %% [markdown]
    # # UCTransNet - ISIC2018
    # ---

    # %% [markdown]
    # # Experiment Logger

    # %%
    #mykey = f'{model_class.__name__}_{key}_{datetime.datetime.now().strftime("%d%b%H")}'
    #import re

    #mykey = re.sub(r'[^a-zA-Z0-9]', '', mykey)
    #mykey += ''.join(random.choices(string.ascii_letters + string.digits, k=33 - len(mykey)))
    #mykey = mykey[:50].ljust(32, '0')
    experiment = Experiment(
        # experiment_key=mykey,
        api_key="8v599AWHmFIfK0YPnIeHWUuwE",
        project_name=dataset_class.__name__,
        workspace="modaresimr",
        log_code=True,
        log_graph=True,
        auto_param_logging=True,  # Can be True or False
        auto_histogram_tensorboard_logging=True,  # Can be True or False
        auto_metric_logging=True  # Can be True or False
    )
    experiment.add_tag(model_class.__name__)
    experiment.add_tag(key)
    # Report multiple hyperparameters using a dictionary:
    # experiment.display(tab='Tab Name')

    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter(comet_config={"disabled": False,'api_key': "8v599AWHmFIfK0YPnIeHWUuwE",
    #   'project_name' : "general",
    #   'workspace':"modaresimr",})

    # %% [markdown]
    # ## Import packages & functions

    # %%

    import os
    import sys
    sys.path.append('../..')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import copy
    import json
    import importlib
    import glob
    import pandas as pd
    from skimage import io, transform
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    import numpy as np
    from tqdm import tqdm

    from utils import (
        show_sbs,
        load_config,
        _print,
    )

    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    # plt.ion()   # interactive mode

    # %% [markdown]
    # ## Set the seed

    # %%
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)

    random.seed(0)

    # %% [markdown]
    # ## Load the config

    # _print("Config:", "info_underline")
    print(json.dumps(config, indent=2))
    print(20 * "~-", "\n")

    # %%

    # %% [markdown]
    # ## Dataset and Dataloader

    # %%
    from datasets.isic import ISIC2018DatasetFast
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    # %%
    # ------------------- params --------------------
    INPUT_SIZE = config['dataset']['input_size']
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ----------------- dataset --------------------
    # preparing training dataset
    tr_dataset = dataset_class(mode="tr", one_hot=True)
    vl_dataset = dataset_class(mode="vl", one_hot=True)
    te_dataset = dataset_class(mode="te", one_hot=True)

    # We consider 1815 samples for training, 259 samples for validation and 520 samples for testing
    # !cat ~/deeplearning/skin/Prepare_ISIC2018.py

    print(f"Length of trainig_dataset:\t{len(tr_dataset)}")
    print(f"Length of validation_dataset:\t{len(vl_dataset)}")
    print(f"Length of test_dataset:\t\t{len(te_dataset)}")

    # prepare train dataloader
    tr_dataloader = DataLoader(tr_dataset, **config['data_loader']['train'])

    # prepare validation dataloader
    vl_dataloader = DataLoader(vl_dataset, **config['data_loader']['validation'])

    # prepare test dataloader
    te_dataloader = DataLoader(te_dataset, **config['data_loader']['test'])

    # -------------- test -----------------
    # test and visualize the input data
    for sample in tr_dataloader:
        img = sample['image']
        msk = sample['mask']
        print("\n Training")
        # show_sbs(img[0], msk[0, 1])
        experiment.log_image(img[0], image_channels="first", name="train/groundtruth")
        break

    for sample in vl_dataloader:
        img = sample['image']
        msk = sample['mask']
        print("Validation")
        # show_sbs(img[0], msk[0, 1])
        experiment.log_image(img[0], image_channels="first", name="val/groundtruth")
        break

    for sample in te_dataloader:
        img = sample['image']
        msk = sample['mask']
        print("Test")
        # show_sbs(img[0], msk[0, 1])
        experiment.log_image(img[0], image_channels="first", name="test/groundtruth")
        break

    # %% [markdown]
    # ### Device

    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch device: {device}")

    # %% [markdown]
    # ## Metrics

    # %%
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(),
            torchmetrics.Accuracy(),
            torchmetrics.Dice(),
            torchmetrics.Precision(),
            torchmetrics.Specificity(),
            torchmetrics.Recall(),
            # IoU
            torchmetrics.JaccardIndex(2)
        ],
        prefix='train_metrics/'
    )

    # train_metrics
    train_metrics = metrics.clone(prefix='train_metrics/').to(device)

    # valid_metrics
    valid_metrics = metrics.clone(prefix='valid_metrics/').to(device)

    # test_metrics
    test_metrics = metrics.clone(prefix='test_metrics/').to(device)

    # %%
    def make_serializeable_metrics(computed_metrics):
        res = {}
        for k, v in computed_metrics.items():
            res[k] = float(v.cpu().detach().numpy())
        return res

    # %% [markdown]
    # ## Define validate function

    # %%
    def validate(model, criterion, vl_dataloader):
        model.eval()
        with torch.no_grad():

            evaluator = valid_metrics.clone().to(device)

            losses = []
            cnt = 0.
            for batch, batch_data in enumerate(vl_dataloader):
                imgs = batch_data['image']
                msks = batch_data['mask']

                cnt += msks.shape[0]

                imgs = imgs.to(device)
                msks = msks.to(device)

                preds = model(imgs)
                loss = criterion(preds, msks)
                losses.append(loss.item())

                preds_ = torch.argmax(preds, 1, keepdim=False).float()
                msks_ = torch.argmax(msks, 1, keepdim=False)
                evaluator.update(preds_, msks_)

    #             _cml = f"curr_mean-loss:{np.sum(losses)/cnt:0.5f}"
    #             _bl = f"batch-loss:{losses[-1]/msks.shape[0]:0.5f}"
    #             iterator.set_description(f"Validation) batch:{batch+1:04d} -> {_cml}, {_bl}")

            # print the final results
            loss = np.sum(losses) / cnt
            metrics = evaluator.compute()

        return evaluator, loss

    # %% [markdown]
    # ## Define train function

    # %%
    def train(
        model,
        device,
        tr_dataloader,
        vl_dataloader,
        config,

        criterion,
        optimizer,
        scheduler,

        save_dir='./',
        save_file_id=None,
    ):

        EPOCHS = tr_prms['epochs']

        torch.cuda.empty_cache()
        model = model.to(device)

        evaluator = train_metrics.clone().to(device)

        epochs_info = []
        best_model = None
        best_result = {}
        best_vl_loss = np.Inf
        for epoch in range(EPOCHS):
            model.train()

            evaluator.reset()
            tr_iterator = tqdm(enumerate(tr_dataloader))
            tr_losses = []
            cnt = 0
            for batch, batch_data in tr_iterator:
                # if batch % 38 != 0:
                #     continue
                imgs = batch_data['image']
                msks = batch_data['mask']

                imgs = imgs.to(device)
                msks = msks.to(device)

                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, msks)
                loss.backward()
                optimizer.step()

                # evaluate by metrics
                preds_ = torch.argmax(preds, 1, keepdim=False).float()
                msks_ = torch.argmax(msks, 1, keepdim=False)
                evaluator.update(preds_, msks_)

                cnt += imgs.shape[0]
                tr_losses.append(loss.item())

                # write details for each training batch
                _cml = f"curr_mean-loss:{np.sum(tr_losses)/cnt:0.5f}"
                _bl = f"mean_batch-loss:{tr_losses[-1]/imgs.shape[0]:0.5f}"
                tr_iterator.set_description(f"Training) ep:{epoch:03d}, batch:{batch+1:04d} -> {_cml}, {_bl}")

            tr_loss = np.sum(tr_losses) / cnt

            # validate model
            vl_metrics, vl_loss = validate(model, criterion, vl_dataloader)
            te_metrics, te_loss = validate(model, criterion, te_dataloader)
            epoch_info = {
                'tr_loss': tr_loss,
                'vl_loss': vl_loss,
                'te_loss': te_loss,
                'tr_metrics': make_serializeable_metrics(evaluator.compute()),
                'vl_metrics': make_serializeable_metrics(vl_metrics.compute()),
                'te_metrics': make_serializeable_metrics(te_metrics.compute())
            }
            if vl_loss < best_vl_loss:
                # find a better model
                best_model = model
                best_vl_loss = vl_loss
                best_result = epoch_info

            print(
                f"trl={tr_loss:0.5f} vll={vl_loss:0.5f}   best_te: loss={best_result['te_loss']:0.5f} acc:{best_result['te_metrics']['valid_metrics/Accuracy']:0.5f} tpr:{best_result['te_metrics']['valid_metrics/Recall']:0.5f} prc:{best_result['te_metrics']['valid_metrics/Precision']:0.5f} f1:{best_result['te_metrics']['valid_metrics/F1Score']:0.5f}")
            # write the final results

            experiment.log_metric('_loss', epoch_info['tr_loss'], epoch=epoch)
            experiment.log_metrics({k.replace("train_", "").replace("metrics/", ""): v for k, v in epoch_info['tr_metrics'].items()}, epoch=epoch)

            # writer.add_scalars(
            #     "losses",
            #     {
            #         "train_loss": epoch_info['tr_loss'],
            #         "val_loss": epoch_info['vl_loss'],
            #         "test_loss": epoch_info['te_loss'],
            #     },
            #     epoch,
            # )

            with experiment.validate():
                experiment.log_metric("_loss", epoch_info['vl_loss'], epoch=epoch)
                experiment.log_metrics({k.replace("valid_", "").replace("metrics/", ""): v for k, v in epoch_info['vl_metrics'].items()}, epoch=epoch)
                # experiment.log_confusion_matrix

            with experiment.test():
                experiment.log_metric("_loss", epoch_info['te_loss'], epoch=epoch)
                experiment.log_metrics({k.replace("valid_", "").replace("metrics/", ""): v for k, v in epoch_info['te_metrics'].items()}, epoch=epoch)

                # experiment.log_metric("best_loss",epoch_info['te_loss'], epoch=epoch)
                experiment.log_metrics({k.replace("valid_", "best_").replace("metrics/", ""): v for k, v in best_result['te_metrics'].items()}, epoch=epoch)

            epochs_info.append(epoch_info)
    #         epoch_tqdm.set_description(f"Epoch:{epoch+1}/{EPOCHS} -> tr_loss:{tr_loss}, vl_loss:{vl_loss}")
            evaluator.reset()

            scheduler.step(vl_loss)

        # save final results
        res = {
            'id': save_file_id,
            'config': config,
            'epochs_info': epochs_info,
            'best_result': best_result
        }
        fn = f"{save_file_id+'_' if save_file_id else ''}result.json"
        fp = os.path.join(config['model']['save_dir'], fn)
        with open(fp, "w") as write_file:
            json.dump(res, write_file, indent=4)

        # save model's state_dict
        fn = "last_model_state_dict.pt"
        fp = os.path.join(config['model']['save_dir'], fn)
        torch.save(model.state_dict(), fp)

        # save the best model's state_dict
        fn = "best_model_state_dict.pt"
        fp = os.path.join(config['model']['save_dir'], fn)
        torch.save(best_model.state_dict(), fp)

        return best_model, model, res

    # %% [markdown]
    # ## Define test function

    # %%
    def test(model, te_dataloader):
        model.eval()
        with torch.no_grad():
            evaluator = test_metrics.clone().to(device)
            for batch_data in tqdm(te_dataloader):
                imgs = batch_data['image']
                msks = batch_data['mask']

                imgs = imgs.to(device)
                msks = msks.to(device)

                preds = model(imgs)

                # evaluate by metrics
                preds_ = torch.argmax(preds, 1, keepdim=False).float()
                msks_ = torch.argmax(msks, 1, keepdim=False)
                evaluator.update(preds_, msks_)
            # experiment.log_confusion_matrix(
            #     msks,
            #     preds_,
            #     images=imgs,
            #     title="Confusion Matrix: Evaluation",
            #     file_name="confusion-matrix-test.json",
            #     image_channels='first'
            # )
        return evaluator

    # %% [markdown]
    # ## Load and prepare model

    # %%
    # download weights

    # !wget "https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz"
    # !mkdir -p ../model/vit_checkpoint/imagenet21k
    # !mv R50+ViT-B_16.npz ../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz

    # %%
    # from models._uctransnet.UCTransNet_ACDA import UCTransNet as Net
    # import models._uctransnet.Config as uct_config
    # config_vit = uct_config.get_CTranS_config()

    # Initialize and train your model
    model = model_class(**config['model']['params'])
    # train(model)

    # Seamlessly log your Pytorch model

    experiment.log_parameters(config)

    torch.cuda.empty_cache()
    model = model.to(device)
    number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", number_of_parameters)
    experiment.log_parameter("Number of parameters", number_of_parameters)
    experiment.set_model_graph(str(model))

    os.makedirs(config['model']['save_dir'], exist_ok=True)
    ali_utils.save_config(config, config['model']['save_dir'] + "/config.yaml")
    model_path = f"{config['model']['save_dir']}/model_state_dict.pt"

    if config['model']['load_weights']:
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained weights...")

    # criterion_dice = DiceLoss()
    criterion_dice = DiceLossWithLogtis()
    # criterion_ce  = BCELoss()
    criterion_ce = CrossEntropyLoss()

    def criterion(preds, masks):
        c_dice = criterion_dice(preds, masks)
        c_ce = criterion_ce(preds, masks)
        return 0.5 * c_dice + 0.5 * c_ce

    tr_prms = config['training']
    optimizer = globals()[tr_prms['optimizer']['name']](model.parameters(), **tr_prms['optimizer']['params'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **tr_prms['scheduler'])

    # %%
    with experiment.train():
        best_model, model, res = train(
            model,
            device,
            tr_dataloader,
            vl_dataloader,
            config,

            criterion,
            optimizer,
            scheduler,

            save_dir=config['model']['save_dir'],
            save_file_id=None,
        )
        log_model(experiment, best_model, model_name="TheModel")

    # %%
    with experiment.test():
        te_metrics = test(best_model, te_dataloader)
        metrics = te_metrics.compute()
        experiment.log_metrics(metrics)
        experiment.end()

        metrics

    # %%
    print(metrics)
    df = pd.DataFrame({k.replace("test_metrics/", ""): v for k, v in metrics.items()}, index=[0])
    from IPython.display import display
    display(df)
    experiment.log_table("result.json", tabular_data=df, headers=True)
    # %%
    f"{config['model']['save_dir']}"

    # %% [markdown]
    # # Test the best inferred model
    # ----

    # %% [markdown]
    # ## Load the best model

    # %%

    best_model = Net(config_vit, **config['model']['params'], num_bases=6)

    torch.cuda.empty_cache()
    best_model = best_model.to(device)

    fn = "best_model_state_dict.pt"
    os.makedirs(config['model']['save_dir'], exist_ok=True)
    model_path = f"{config['model']['save_dir']}/{fn}"

    best_model.load_state_dict(torch.load(model_path))
    print("Loaded best model weights...")

    # %% [markdown]
    # ## Evaluation

    # %%
    te_metrics = test(best_model, te_dataloader)
    te_metrics.compute()

    # %% [markdown]
    # ## Plot graphs

    # %%
    result_file_path = f"{config['model']['save_dir']}/result.json"
    with open(result_file_path, 'r') as f:
        results = json.loads(''.join(f.readlines()))
    epochs_info = results['epochs_info']

    tr_losses = [d['tr_loss'] for d in epochs_info]
    vl_losses = [d['vl_loss'] for d in epochs_info]
    tr_dice = [d['tr_metrics']['train_metrics/Dice'] for d in epochs_info]
    vl_dice = [d['vl_metrics']['valid_metrics/Dice'] for d in epochs_info]
    tr_js = [d['tr_metrics']['train_metrics/JaccardIndex'] for d in epochs_info]
    vl_js = [d['vl_metrics']['valid_metrics/JaccardIndex'] for d in epochs_info]
    tr_acc = [d['tr_metrics']['train_metrics/Accuracy'] for d in epochs_info]
    vl_acc = [d['vl_metrics']['valid_metrics/Accuracy'] for d in epochs_info]

    _, axs = plt.subplots(1, 4, figsize=[16, 3])

    axs[0].set_title("Loss")
    axs[0].plot(tr_losses, 'r-', label="train loss")
    axs[0].plot(vl_losses, 'b-', label="validatiton loss")
    axs[0].set_ylim([0, 0.015])
    axs[0].legend()

    axs[1].set_title("Dice score")
    axs[1].plot(tr_dice, 'r-', label="train dice")
    axs[1].plot(vl_dice, 'b-', label="validation dice")
    axs[1].set_ylim([0.77, 1])
    axs[1].legend()

    axs[2].set_title("Jaccard Similarity")
    axs[2].plot(tr_js, 'r-', label="train JaccardIndex")
    axs[2].plot(vl_js, 'b-', label="validatiton JaccardIndex")
    axs[2].set_ylim([0.75, 1])
    axs[2].legend()

    axs[3].set_title("Accuracy")
    axs[3].plot(tr_acc, 'r-', label="train Accuracy")
    axs[3].plot(vl_acc, 'b-', label="validation Accuracy")
    axs[3].set_ylim([0.9, 1])
    axs[3].legend()

    plt.show()

    # %%
    epochs_info

    # %% [markdown]
    # ## Save images

    # %%
    from PIL import Image
    import cv2

    def skin_plot(img, gt, pred):
        img = np.array(img)
        gt = np.array(gt)
        pred = np.array(pred)
        edged_test = cv2.Canny(pred, 100, 255)
        contours_test, _ = cv2.findContours(edged_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        edged_gt = cv2.Canny(gt, 100, 255)
        contours_gt, _ = cv2.findContours(edged_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt_test in contours_test:
            cv2.drawContours(img, [cnt_test], -1, (0, 0, 255), 1)
        for cnt_gt in contours_gt:
            cv2.drawContours(img, [cnt_gt], -1, (0, 255, 0), 1)
        return img

    # ---------------------------------------------------------------------------------------------

    save_imgs_dir = f"{config['model']['save_dir']}/visualized"

    if not os.path.isdir(save_imgs_dir):
        os.mkdir(save_imgs_dir)

    with torch.no_grad():
        for batch in tqdm(te_dataloader):
            imgs = batch['image']
            msks = batch['mask']
            ids = batch['id']

            preds = best_model(imgs.to(device))

            txm = imgs.cpu().numpy()
            tbm = torch.argmax(msks, 1).cpu().numpy()
            tpm = torch.argmax(preds, 1).cpu().numpy()
            tid = ids

            for idx in range(len(tbm)):
                img = np.moveaxis(txm[idx, :3], 0, -1) * 255.
                img = np.ascontiguousarray(img, dtype=np.uint8)
                gt = np.uint8(tbm[idx] * 255.)
                pred = np.where(tpm[idx] > 0.5, 255, 0)
                pred = np.ascontiguousarray(pred, dtype=np.uint8)

                res_img = skin_plot(img, gt, pred)

                fid = tid[idx]
                Image.fromarray(img).save(f"{save_imgs_dir}/{fid}_img.png")
                Image.fromarray(res_img).save(f"{save_imgs_dir}/{fid}_img_gt_pred.png")

    # %%
    f"{config['model']['save_dir']}/visualized"

    from PIL import Image
    import cv2

    def skin_plot(img, gt, pred):
        img = np.array(img)
        gt = np.array(gt)
        pred = np.array(pred)
        edged_test = cv2.Canny(pred, 100, 255)
        contours_test, _ = cv2.findContours(edged_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        edged_gt = cv2.Canny(gt, 100, 255)
        contours_gt, _ = cv2.findContours(edged_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt_test in contours_test:
            cv2.drawContours(img, [cnt_test], -1, (0, 0, 255), 1)
        for cnt_gt in contours_gt:
            cv2.drawContours(img, [cnt_gt], -1, (0, 255, 0), 1)
        return img

    # ---------------------------------------------------------------------------------------------

    save_imgs_dir = f"{config['model']['save_dir']}/visualized"

    if not os.path.isdir(save_imgs_dir):
        os.mkdir(save_imgs_dir)

    with torch.no_grad():
        for batch in tqdm(te_dataloader):
            imgs = batch['image']
            msks = batch['mask']
            ids = batch['id']

            preds = best_model(imgs.to(device))

            txm = imgs.cpu().numpy()
            tbm = torch.argmax(msks, 1).cpu().numpy()
            tpm = torch.argmax(preds, 1).cpu().numpy()
            tid = ids

            for idx in range(len(tbm)):
                img = np.moveaxis(txm[idx, :3], 0, -1) * 255.
                img = np.ascontiguousarray(img, dtype=np.uint8)
                gt = np.uint8(tbm[idx] * 255.)
                pred = np.where(tpm[idx] > 0.5, 255, 0)
                pred = np.ascontiguousarray(pred, dtype=np.uint8)

                res_img = skin_plot(img, gt, pred)

                fid = tid[idx]
                Image.fromarray(img).save(f"{save_imgs_dir}/{fid}_img.png")
                Image.fromarray(res_img).save(f"{save_imgs_dir}/{fid}_img_gt_pred.png")
