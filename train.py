import sys
sys.path.append('/mnt/workplace/blurHand')
import torch
from tqdm import tqdm
from net.HMR.config import cfg
# from net.SAR.config import cfg
from base import Trainer
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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    trainer = Trainer()
    trainer._make_model()
    trainer._make_batch_loader()
    prev_lr = cfg.lr
    cur_lr = cfg.lr
    best_loss = np.inf
    torch.cuda.empty_cache()
    gc.collect()
    summary_folder = os.path.join(cfg.base_folder, 'runs')
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)
    writer = SummaryWriter(summary_folder, filename_suffix="_" + cfg.dataset)
    for epoch in range(trainer.start_epoch, cfg.total_epoch):
        loss_dict = {}
        loss_dict['total_loss'] = 0
        for k in cfg.loss_queries:
            loss_dict[k] = 0
        #####Start One-epoch
        for iteration, (sample) in tqdm(enumerate(trainer.train_loader), total=len(trainer.train_loader)):
            trainer.optimizer.zero_grad()
            input = sample['image']
            loss = trainer.model(input, target=sample)
            for k, v in loss.items():
                v_mean = v.mean()
                loss[k] = v_mean
                loss_dict[k] += v_mean.detach().cpu()
            if cfg.loss_weight is not None:
                loss_sum = sum(loss[k] * cfg.loss_weight[k] for k in cfg.loss_queries)
            else:
                loss_sum = sum(loss[k] for k in cfg.loss_queries)
            loss_dict['total_loss'] += loss_sum
            loss_sum.backward()
            trainer.optimizer.step()
            if (iteration+1) % cfg.print_iter == 0:
                screen = ['[Epoch %d/%d]' % (epoch, cfg.total_epoch), '[Batch %d/%d]' % (iteration+1, len(trainer.train_loader))]
                screen += ['[%s: %.4f]' % ('loss_' + k, v.detach()) for k, v in loss.items()]
                trainer.logger.info(''.join(screen))
        #####End One-epoch
        for k, v in loss_dict.items():
            loss_dict[k]  = v / len(trainer.train_loader)
            writer.add_scalar('[Becnchmark][%s on %s] %s_loss'%(cfg.pre, cfg.dataset, k), loss_dict[k], epoch)
            writer.flush()
            writer.close()
        trainer.schedule.step()
        cur_lr = trainer.optimizer.param_groups[0]['lr']
        if cur_lr != prev_lr:
            trainer.logger.info('Learning rate %.2e => %.2e'%(prev_lr, cur_lr))
            prev_lr = cur_lr
        if loss_dict['total_loss'].data.item() < best_loss:
            trainer.save_model(trainer.model, trainer.optimizer, trainer.schedule, epoch)
        
if __name__ == '__main__':
    main()