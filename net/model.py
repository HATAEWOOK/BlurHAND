import sys
import torch
import torch.nn
from SAR.model import SAR
from HMR.hmr_sv import HMR_SV

def get_model(model_name):
    if model_name == "SAR":
        return SAR()
    elif model_name == "HMR_SV":
        return HMR_SV()

if __name__ == '__main__':
    model = get_model("HMR_SV")
    print(model)