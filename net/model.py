import sys
sys.path.append('/root/workplace/backup/blurHand')
import torch
import torch.nn
from net.SAR.model import SAR
from net.HMR.hmr_sv import HMR_SV
from net.evalNet.model import resnet34_Mano

def get_model(model_name):
    if model_name == "SAR":
        return SAR()
    elif model_name == "HMR_SV":
        return HMR_SV()
    elif model_name == "evalNet":
        return resnet34_Mano()

if __name__ == '__main__':
    model = get_model("evalNet")
    print(model)