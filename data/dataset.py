from shutil import ExecError
import sys
import json
import os
import numpy as np
import traceback
import random
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import cv2
import skimage.io as io
import glob

def get_dataset(dat_name, base_path, set_name = 'training'):
    if dat_name == 'FreiHAND':
        return FreiHAND(base_path, set_name)

class FreiHAND(Dataset):
    def __init__(self, base_path, set_name):
        self.base_path = base_path
        self.set_name = set_name
        self.totensor = torchvision.transforms.ToTensor()
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize([224,224]), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(*mean_std)])
        self.load_dataset()
        self.name = "FreiHAND"

    def load_dataset(self):
        self.K_list = json_load(os.path.join(self.base_path, '%s_K.json'%self.set_name))
        self.scale_list = json_load(os.path.join(self.base_path, '%s_scale.json'%self.set_name))
        idxs = sorted([int(imgname.split(".")[0]) for imgname in os.listdir(os.path.join(self.base_path, self.set_name, 'rgb'))])
        self.prefixs = ["%08d"%idx for idx in idxs]
        del idxs
        
        if self.set_name == 'training':
            self.verts_list = json_load(os.path.join(self.base_path, 'training_verts.json'))
            self.j3d_list = json_load(os.path.join(self.base_path, 'training_xyz.json'))

    def get_sample(self, idx):
        sample = {}
        image = self.get_img(idx)
        sample['image'] = self.transform(image)
        K = self.get_K(idx)
        M = torch.FloatTensor([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        sample['Ks'] = K
        sample['Ms'] = M
        sample['scale'] = self.hand_dataset.get_scale(idx)
        if self.set_name == 'training':
            j3d = self.hand_dataset.get_j3d(idx)
            sample['j3d'] = j3d
            verts = self.hand_dataset.get_verts(idx)
            sample['vert'] = verts
            mask = self.hand_dataset.get_mask(idx)
            sample['mask'] = torch.round(self.totensor(mask))
        sample['idx'] = idx
        sample['name'] = self.hand_dataset.name
        img_idx = self.hand_dataset.get_filename(idx)
        sample['filename'] = img_idx

    def __getitem__(self, idx):
        try:
            sample = self.get_sample(idx)
        except ExecError:
            raise "Error at {}".format(idx)
        return sample

    def get_img(self, idx):
        img_path = os.path.join(self.base_path, self.set_name, 'rgb', '{}.jpg'.format(self.prefixs[idx]))
        img = Image.open(img_path).convert('RGB')
        return img

    def get_cv_img(self, idx):
        img_path = os.path.join(self.base_path, self.set_name, 'rgb', '{}.jpg'.format(self.prefixs[idx]))
        cv_img = cv2.imread(img_path)
        return cv_img

    def get_mask(self, idx):
        if idx >= 32560:
            idx = idx % 32560
        mask_path = os.path.join(self.base_path, self.set_name, 'mask', '{}.jpg'.format(self.prefixs[idx]))
        mask = Image.open(mask_path)
        return mask

    def get_maskRGB(self, idx):
        img_path = os.path.join(self.base_path, self.set_name, 'rgb', '{}.jpg'.format(self.prefixs[idx]))
        img = io.imread(img_path)
        if idx >= 32560:
            idx = idx % 32560
        mask_path = os.path.join(self.base_path, self.set_name, 'mask', '{}.jpg'.format(self.prefixs[idx]))
        mask_img = io.imread(mask_path, 1)
        mask_img = np.rint(mask_img)
        img[mask_img < 200] = 0
        return img
        
    def get_K(self, idx):
        if idx >= 32560:
            idx = idx % 32560
        K = torch.FloatTensor(np.array(self.K_list[idx]))
        return K

    def get_scale(self, idx):
        if idx >= 32560:
            idx = idx % 32560
        scale = self.scale_list[idx]
        return scale

    def get_j3d(self, idx):
        if idx >= 32560:
            idx = idx % 32560
        joint = torch.FloatTensor(self.j3d_list[idx])
        return joint

    def get_verts(self, idx):
        if idx >= 32560:
            idx = idx % 32560
        verts = torch.FloatTensor(self.verts_list[idx])
        return verts

    def get_filename(self, idx):
        img_idx = os.path.join(self.base_path, self.set_name, 'rgb', '{}.jpg'.format(self.prefixs[idx]))
        return img_idx

    def __len__(self):
        return len(self.prefixs)

def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d