import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torch.nn.functional as F


class SegPC2021Dataset(Dataset):
    def __init__(self,
                 mode,  # 'tr'-> train, 'vl' -> validation, 'te' -> test
                 input_size=224,
                 scale=2.5,
                 data_dir=None,
                 dataset_dir=None,
                 one_hot=True,
                 force_rebuild=False,
                 img_transform=None,
                 msk_transform=None, cytoplasm=True, nucleus=True, **kwargs):
        # pre-set variables
        self.data_dir = data_dir if data_dir else "/raid/home/labusermodaresi/datasets/TCIA_SegPC_dataset/np"
        self.dataset_dir = dataset_dir if dataset_dir else "/raid/home/labusermodaresi/datasets/TCIA_SegPC_dataset/"
        self.mode = mode
        # input parameters
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.input_size = input_size
        self.scale = scale
        self.one_hot = one_hot
        self.number_classes = nucleus + cytoplasm + 1
        self.nucleus = nucleus
        self.cytoplasm = cytoplasm
        # loading data
        self.load_dataset(force_rebuild=force_rebuild)

    def load_dataset(self, force_rebuild=False):
        INPUT_SIZE = self.input_size
        ADD = self.data_dir

#         build_segpc_dataset(
#             input_size = self.input_size,
#             scale = self.scale,
#             data_dir = self.data_dir,
#             dataset_dir = self.dataset_dir,
#             mode = self.mode,
#             force_rebuild = force_rebuild,
#         )

        print(f'loading X_{self.mode}...')
        self.X = np.load(f'{ADD}/cyts_{self.mode}_{self.input_size}x{self.input_size}_s{self.scale}_X.npy')
        print(f'loading Y_{self.mode}...')
        self.Y = np.load(f'{ADD}/cyts_{self.mode}_{self.input_size}x{self.input_size}_s{self.scale}_Y.npy')
        print(f'finished.cyts_{self.mode}_{self.input_size}x{self.input_size}_s{self.scale}_Y.npy')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        msk1 = self.Y[idx]
        if self.cytoplasm and not self.nucleus:
            msk = np.where((msk1 < 200) & (msk1 > 0.5), 1, 0)
        elif self.nucleus and not self.cytoplasm:
            msk = np.where(msk1 < 200, 0, 1)
        else:
            msk = np.uint8(np.where(msk1 < 0.5, 0, 1) + np.where(msk1 < 200, 0, 1))

        # print(np.max((msk)))
        # np.save('c.npy', msk1)
        if self.img_transform:
            img = self.img_transform(img)
            img = (img - img.min()) / (img.max() - img.min())
        # np.save('b.npy', msk)

        if self.msk_transform:
            msk = self.msk_transform(msk)
            # msk = (msk - msk.min())/(msk.max() - msk.min())
        # np.save('a.npy', msk)

        # print('s', msk.shape)

        if self.one_hot:
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64), self.number_classes)
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)
            # print(msk.shape, 'ssss')
            # msk[1, :, :] += msk[2, :, :]

        sample = {'image': img, 'mask': msk, 'id': idx}
        return sample
