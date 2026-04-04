"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import numpy as np
import cv2
#import matplotlib.pyplot as plt
from data.base_dataset import BaseDataset, get_transform,get_params, get_transform_A1
from data.image_folder import make_dataset
from PIL import Image
import scipy.io as sio
import h5py
import os
import matplotlib.pyplot as plt
import torch
import scipy.io as scio
import xarray as xr
class TemplateDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=100000000, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        print(self.AB_paths)
        print(self.dir_AB)
        assert (self.opt.load_size >= self.opt.crop_size)
    def __getitem__(self, index):

        AB_path = self.AB_paths[index]
        GF_simu=1
        rot90 = 0
        log_nor_option = 0 # 是否对输出海浪谱进行log归一化，0代表0-1归一化
        input_SAR_nor =1 #判断是否对输入SAR图像谱 归一化
        kx=np.linspace(-0.256, 0.254,256)
        ky=kx
        ky_grid,kx_grid=np.meshgrid(ky,kx)
        k=np.sqrt(np.power(kx_grid,2)+np.power(ky_grid,2))
        if GF_simu==1:
            AB = h5py.File(AB_path, 'r')
            A_1 = np.array(AB['simu_spec_vv'][:]).astype(np.float32)  # 读取VV
            #B_1 = np.array(AB['era_f_phi'][:]).astype(np.float32)
            B_1 = np.array(AB['era_spec'][:]).astype(np.float32)
            A_3 = np.array(AB['incidence'][:]).astype(np.float32)
            A_2 = np.array(AB['beta_used'][:]).astype(np.float32)
            A_2 = np.ones((256, 256), dtype=np.float32) * A_2
            A_3 = np.ones((256, 256), dtype=np.float32) * A_3
        if rot90 == 1:
            A_1 = np.rot90(A_1)
            A_2 = np.rot90(A_2)
            B_1 = np.rot90(B_1)
        if log_nor_option==1:
            B_1 = np.log10(B_1 + 1) / 4
        else :
            B_1 = cv2.normalize(B_1, None, 0, 1, cv2.NORM_MINMAX)  # 对输出海浪谱进行单个归一化
        if input_SAR_nor==1:
            A_1 = cv2.normalize(A_1, None, 0, 1, cv2.NORM_MINMAX)
        A_1=A_1.reshape(256,256,-1)
        ##### normalized the incidence angle
        min_inc = 0.37
        A_3 = (A_3 - min_inc) / (0.7 - 0.37)
        A_3 = A_3 * np.ones((import_SIZE, import_SIZE), dtype=np.float32)
        A_1 = A_1.reshape(256, 256, -1)
        A_3 = A_3.reshape(256, 256, -1)
        A_input = np.concatenate((A_1, A_3), axis=2)

        B_nor_resize = B_1.reshape(256, 256,-1)
        ### 让数据正态分布，使其便于训练。
        transform_params = get_params(self.opt, A_input.shape)
        A_transform = get_transform_A1(self.opt, transform_params)
        # not use normalize
        B_transform = get_transform(self.opt, transform_params)
        A = A_transform(A_input) #应用归一化
        B = B_transform(B_nor_resize)
        #B=np.nan_to_num(B)
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
    def __len__(self):
        """Return the total number of images."""
        return len(self.AB_paths)
