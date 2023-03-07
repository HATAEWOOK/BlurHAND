import sys
sys.path.append('/mnt/workplace/blurHand')
import torch
from tqdm import tqdm
from net.HMR.config import cfg
# from net.SAR.config import cfg
from base import Tester
import numpy as np
import os
import random
from datetime import datetime
import gc
from torch.utils.tensorboard import SummaryWriter

def main():
    torch.cuda.manual_seed_all(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    random.seed(cfg.manual_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    tester = Tester()
    tester._make_model()
    tester._make_batch_loader()
    mpjpe = []
    for iteration, (sample) in tqdm(enumerate(tester.valid_loader), total=len(tester.valid_loader)):
        with torch.no_grad():
            input = sample['image']
            outs, eval_matrix = tester.model(input, target=sample)
            tester.visualization(outs, iteration, input)
            mpjpe.append(eval_matrix['MPJPE'].item())
            tester.logger.info(sum(mpjpe) / len(mpjpe))

    tester.logger.info(sum(mpjpe) / len(mpjpe))

if __name__ == '__main__':
    main()