import numpy as np
from utils.util_loss import normalize
import torch

def get_mpjpe(outs, tars): #[bs, 21, 3] , [bs, 21, 3]
    out_norm = normalize(outs)
    tar_norm = normalize(tars)
    dis = torch.norm((tar_norm - out_norm), dim = 2)
    mpjpe = torch.mean(dis)
    return mpjpe