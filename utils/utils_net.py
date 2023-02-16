import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import cv2
import torch
import torch.nn.functional
import torch.nn as nn
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import torch.cuda.comm

def normalize_image(im):
    """
    byte -> float, / pixel_max, - 0.5
    :param im: torch byte tensor, B x C x H x W, 0 ~ 255
    :return:   torch float tensor, B x C x H x W, -0.5 ~ 0.5
    """
    return ((im.float()) - 0.5)
    # return ((im.float() / 255.0) - 0.5)

#heatmap -> uv
def compute_uv_from_integral(hm, resize_dim):
    """
    https://github.com/JimmySuen/integral-human-pose
    
    :param hm: B x K x H x W (Variable)
    :param resize_dim:
    :return: uv in resize_dim (Variable)
    
    heatmaps: C x H x W
    return: C x 3
    """
    upsample = nn.Upsample(size=resize_dim, mode='bilinear', align_corners=True)  # (B x K) x H x W
    resized_hm = upsample(hm).view(-1, resize_dim[0], resize_dim[1]) #[bs*21, 256, 256]
    #import pdb; pdb.set_trace()
    num_joints = resized_hm.shape[0] #bs*21
    hm_width = resized_hm.shape[-1] #256
    hm_height = resized_hm.shape[-2] #256
    hm_depth = 1
    pred_jts = softmax_integral_tensor(resized_hm, num_joints, hm_width, hm_height, hm_depth) #[1,2016]
    pred_jts = pred_jts.view(-1,hm.size(1), 3)
    #import pdb; pdb.set_trace()
    return pred_jts #[bs, 21, 3]

def softmax_integral_tensor(preds, num_joints, hm_width, hm_height, hm_depth):
    # global soft max
    #preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = preds.reshape((1, num_joints, -1)) #[1, bs*21, 65536]
    preds = torch.nn.functional.softmax(preds, 2) #[1, bs*21, 65536]
    # integrate heatmap into joint location
    x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
    #x = x / float(hm_width) - 0.5
    #y = y / float(hm_height) - 0.5
    #z = z / float(hm_depth) - 0.5
    preds = torch.cat((x, y, z), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 3))
    return preds

def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)
    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))#[1,B*21,1,height,width]
    accu_x = heatmaps.sum(dim=2)
    accu_x = accu_x.sum(dim=2)#[1,B*21,width=256]
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)#[1,B*21,hight=256]
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)#[1,B*21,depth=1]
    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(z_dim).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]
    accu_x = accu_x.sum(dim=2, keepdim=True) #[1,672,1]
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)
    #import pdb; pdb.set_trace()
    return accu_x, accu_y, accu_z
##

def mano2Frei(Mano_joints):#[b,21,3]
    FreiHand_joints = torch.zeros_like(Mano_joints).to(Mano_joints.device) # init empty list

    # manoId, FreiId
    mapping = {0: 0, #Wrist
               1: 5, 2: 6, 3: 7, 17: 8, #Index
               4: 9, 5: 10, 6: 11, 18: 12, #Middle
               7: 17, 8: 18, 9: 19, 20: 20, # Pinky
               10: 13, 11: 14, 12: 15, 19: 16, # Ring
               13: 1, 14: 2, 15: 3, 16: 4,} # Thumb

    for manoId, myId in mapping.items():
        FreiHand_joints[:,myId] = Mano_joints[:,manoId]
    #import pdb; pdb.set_trace()
    return FreiHand_joints


def proj_func(xyz, K):
    '''
    xyz: N x num_points x 3
    K: N x 3 x 3
    '''
    uv = torch.bmm(K,xyz.permute(0,2,1))
    uv = uv.permute(0, 2, 1)
    out_uv = torch.zeros_like(uv[:,:,:2]).to(device=uv.device)
    out_uv = torch.addcdiv(out_uv, uv[:,:,:2], uv[:,:,2].unsqueeze(-1).repeat(1,1,2), value=1)
    return out_uv
    
def orthographic_proj_withz(X, trans, scale, offset_z=0.):
    """
    X: B x N x 3
    trans: B x 2: [tx, ty]
    scale: B x 1: [sc]
    Orth preserving the z.
    """
    scale = scale.contiguous().view(-1, 1, 1)
    trans = trans.contiguous().view(scale.size(0), 1, -1)
    proj = scale * X

    proj_xy = proj[:, :, :2] + trans[:,:,:2]
    proj_z = proj[:, :, 2, None] + offset_z
    return torch.cat((proj_xy, proj_z), 2)