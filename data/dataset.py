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
import pickle
from utils import augmentation
from utils.util_loss import normalize

def get_dataset(dat_name, base_path, set_name = 'training'):
    if dat_name == 'FreiHAND':
        return FreiHAND(base_path, set_name)
    
    if dat_name == 'blurHand':
        return blurHand(base_path)
    
    if dat_name == 'RHD':
        return RHD(base_path, set_name)

class FreiHAND(Dataset):
    def __init__(self, base_path, set_name):
        self.base_path = base_path
        self.set_name = set_name
        self.totensor = torchvision.transforms.ToTensor()
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize([224,224]), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(*mean_std)])
        self.load_dataset()
        self.name = "FreiHAND"
        self.max_rot = np.pi

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
        center = np.asarray([112,112])
        scale = 224
        rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
        rot_mat = np.array(
        [
            [np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot), np.cos(rot), 0],
            [0, 0, 1],
        ]
        ).astype(np.float32)
        affinetrans, post_rot_trans = augmentation.get_affine_transform(
            center, scale, [224, 224], rot=rot
        )
        trans_image = augmentation.transform_img(
            image, affinetrans, [224, 224]
        )
        trans_img = self.totensor(trans_image).float()
        sample['image'] = trans_img
        K = self.get_K(idx)
        trans_Ks = post_rot_trans.dot(K)
        M = torch.FloatTensor([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        sample['Ks'] = trans_Ks
        sample['Ms'] = M
        sample['scale'] = self.get_scale(idx)
        if self.set_name == 'training':
            j3d = self.get_j3d(idx)
            trans_j3d = rot_mat.dot(
                        j3d.transpose(1,0)
                    ).transpose()
            sample['j3d'] = trans_j3d
            verts = self.get_verts(idx)
            trans_verts = rot_mat.dot(
                        verts.transpose(1,0)
                    ).transpose()
            sample['vert'] = trans_verts
            mask = self.get_mask(idx)
            trans_masks = augmentation.transform_img(
                        mask, affinetrans, [224,224]
                    )
            trans_mask = torch.round(self.totensor(trans_masks))
            sample['mask'] = trans_mask
        sample['idx'] = idx
        sample['name'] = self.name
        img_idx = self.get_filename(idx)
        sample['filename'] = img_idx

        return sample

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
    
class blurHand(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        self.totensor = torchvision.transforms.ToTensor()
        mean_std = ([0.4532, 0.4522, 0.4034], [0.2485, 0.2418, 0.2795])
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize([224,224]), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(*mean_std)])
        self.load_dataset()
        self.name = "blurHand"

    def load_dataset(self):
        self.K_sharp_list = json_load(os.path.join(self.base_path, 'calib_k.json'))['mas']
        self.K_blur_list = json_load(os.path.join(self.base_path, 'calib_k.json'))['sub1']
        self.ext_sharp_list = json_load(os.path.join(self.base_path, 'calib_ext.json'))['mas']
        self.ext_blur_list = json_load(os.path.join(self.base_path, 'calib_ext.json'))['sub1']
        self.mano_list = json_load(os.path.join(self.base_path, 'mano.json'))
        self.xyz_list = json_load(os.path.join(self.base_path, 'xyz.json'))
        self.mp_keypt_list = json_load(os.path.join(self.base_path, 'mp_mas.json'))

    def get_sample(self, idx):
        sample={}
        simg = self.get_sharp_img(idx)
        sample['image'] = self.totensor(simg).float()
        bimg = self.get_blur_img(idx)
        sample['bimage'] = self.totensor(bimg).float()
        sample['Ks'] = self.get_sharp_K(idx)
        sample['Ext'] = self.get_sharp_ext(idx)
        sample['bKs'] = self.get_blur_K(idx)
        sample['bExt'] = self.get_blur_ext(idx)
        sample['j3d'] = self.get_xyz(idx)
        sample['mask'] = torch.round(self.totensor(self.get_mask(idx)))
        sample['idx'] = idx
        sample['name'] = self.name

        return sample
    
    def __getitem__(self, idx):
        try:
            sample = self.get_sample(idx)
        except ExecError:
            raise "Error at {}".format(idx)
        return sample

    def get_sharp_img(self, idx):
        img = Image.open(os.path.join(self.base_path, 'rgb', 'mas', '%05d.jpg'%idx)).convert('RGB')
        return img
    
    def get_blur_img(self, idx):
        img = Image.open(os.path.join(self.base_path, 'rgb', 'sub1', '%05d.jpg'%idx)).convert('RGB')
        return img
    
    def get_mask(self, idx):
        mask = Image.open(os.path.join(self.base_path, 'mask', 'mas', '%05d.jpg'%idx))
        return mask
    
    def get_sharp_K(self, idx):
        K = self.K_sharp_list[idx]
        return torch.FloatTensor(K)
    
    def get_blur_K(self, idx):
        K = self.K_blur_list[idx]
        return torch.FloatTensor(K)
    
    def get_sharp_ext(self, idx):
        ext = self.ext_sharp_list[idx]
        return torch.FloatTensor(ext)
    
    def get_blur_ext(self, idx):
        ext = self.ext_blur_list[idx]
        return torch.FloatTensor(ext)
    
    def get_mano(self, idx):
        mano = self.mano_list[idx]
        return torch.FloatTensor(mano)
    
    def get_xyz(self, idx):
        xyz = self.xyz_list[idx]
        return torch.FloatTensor(xyz)
    
    def get_keypt(self, idx):
        keypt = self.mp_keypt_list[idx]
        return torch.FloatTensor(keypt)
    
    def __len__(self):
        return len(self.xyz_list)
    
class RHD(Dataset):
    def __init__(self, base_path, set_name):
        self.base_path = base_path
        self.set_name = set_name
        self.totensor = torchvision.transforms.ToTensor()
        mean_std = None #Todo
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize([224,224]), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(*mean_std)])
        self.load_dataset()
        self.name = "RHD"

    def load_dataset(self):
        with open(os.path.join(self.base_path, self.set_name, 'anno_%s.pickle' % self.set_name), 'rb') as fi:
            self.anno_all = pickle.load(fi)

    def get_sample(self, idx):
        sample={}
        sample['image'] = None
        sample['Ks'] = None
        sample['j3d'] = None
        sample['keypt'] = None
        sample['mask'] = None
        sample['idx'] = idx
        sample['name'] = self.name


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d