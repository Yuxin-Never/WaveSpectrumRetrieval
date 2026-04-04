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
from data.base_dataset import BaseDataset, get_transform,get_params
from data.image_folder import make_dataset
from PIL import Image
import scipy.io as sio
import h5py
import os
import matplotlib.pyplot as plt
import torch
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
        parser.set_defaults(max_dataset_size=10000, new_dataset_option=2.0)  # specify dataset-specific default values
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
        #self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        #self.transform = get_transform(opt)
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB=h5py.File(AB_path,'r')
        A_1=np.array(AB['simu_vv_256'][:]).astype(np.float32) #读取VV
        A_2=np.array(AB['simu_hh_256'][:]).astype(np.float32) #读取HH
        A_1 = np.array(AB['VV_spec'][:]).astype(np.float32)  # 读取VV
        A_2 = np.array(AB['HH_spec'][:]).astype(np.float32)  # 读取HH
        A_3 = np.array(AB['inc_256'][:]).astype(np.float32) #读取入射角
        cv2.normalize(A_1, A_1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(A_2, A_2, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(A_3, A_3, 0, 1, cv2.NORM_MINMAX)
        A_2=cv2.blur((abs(A_2)), (2, 2))
        A_1 = cv2.blur((abs(A_1)), (2, 2))
        A_2 = np.rot90(A_2)
        A_1 = np.rot90(A_1)
        A_2 = A_2[:,::-1]
        A_1 = A_1[:, ::-1]
        A_1=A_1.reshape(256,256,-1)
        A_2=A_2.reshape(256,256,-1)
        A_3=A_3.reshape(256, 256, -1)
        A=np.concatenate((A_1,A_2),axis=2)
        A = np.concatenate((A, A_3), axis=2)
        #A = A_1
        #print(A.shape)
        #fig,ax=plt.subplots()
        #ax.contourf(s[:,:,0])
        #plt.show()
        #print(A.shape)
        #A=np.abs(A) #need to set as float 32
        #A=A
        #print('Ashape',A.type)
        B_1=np.array(AB['era_256'][:]).astype(np.float32)
        cv2.normalize(B_1, B_1, 0, 1, cv2.NORM_MINMAX)
        B_1 = B_1.reshape(256, 256,-1)
        B=B_1
        #print(B)
        #B=np.concatenate((B_1, B_1), axis=2)
        #B = np.concatenate((B, B_1), axis=2)
        #np.dstack
        #print(B.shape)
        #A=np.reshape(A,(A.shape[0],A.shape[1],1))
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.shape)
        A_transform = get_transform(self.opt, transform_params)
        # not use normalize
        B_transform = get_transform(self.opt, transform_params)

        #ImgB=Image.fromarray(B)
        #print(ImgA.size)
        #size=256
        #A = torch.tensor(A)
        #print(A.shape)
        #B = torch.tensor(B)
        A = A_transform(A)
        B = B_transform(B)
        #print(A.shape,'single')
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images."""

        return len(self.AB_paths)
