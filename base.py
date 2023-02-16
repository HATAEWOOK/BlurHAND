#Source : https://github.com/zxz267/SAR 
import torch
import torch.nn as nn
import yaml
import os
from torch.utils.data import DataLoader
from datetime import datetime
import torch.optim as optim
from config import cfg
from utils.logger import setup_logger
from data.dataset import get_dataset
from net.model import get_model

class Trainer:
    def __init__(self):
        cfg.output_root = './train/%s'%cfg.dataset
        today = datetime.strftime(datetime.now().replace(microsecond=0), '%Y-%m-%d')
        base_folder = os.path.join(cfg.output_root, "%s_%d"%(today, cfg.trial_num))
        if os.path.exists(base_folder):
            cfg.trial_num += 1
            base_folder = os.path.join(cfg.output_root, "%s_%d"%(today, cfg.trial_num))
            os.makedirs(base_folder)
            cfg.base_folder = base_folder
        
        log_folder = os.path.join(base_folder, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'train_' + cfg.experiment_name + '.log')
        with open(os.path.join(log_folder, 'train_' + cfg.experiment_name + '.yaml')) as fw:
            yaml.safe_dump(vars(cfg), fw, default_flow_style=False)

        self.logger = setup_logger(output=logfile, name='%s_training'%cfg.pre)
        self.logger.info('Start training : %s' % ('train_' + cfg.experiment_name))

    def get_optimizer(self, model):
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=0.001)
        self.logger.info('Adam optimizer / Initial learning rate {}'.format(cfg.lr))
        return optimizer

    def get_schedule(self, optimizer):
        schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.total_epoch, eta_min=0)
        self.logger.info('Sceduler / CosineAnnealingLR')
        return schedule

    def load_model(self, model, optimizer, schedule):
        if cfg.checkpoint is not None:
            checkpoint = torch.load(cfg.checkpoint)
            self.logger.info('Load model from {}'.format(cfg.checkpoint))
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            schedule.load_state_dict(checkpoint['schedule'])
            start_epoch = checkpoint['last_epoch'] + 1
            self.logger.info('Model loaded')
        return start_epoch, model

    def save_model(self, model, optimizer, schedule, epoch):
        save = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'schedule': schedule.state_dict(),
            'last_epoch': epoch
        }
        path_checkpoint = os.path.join(cfg.base_folder , 'checkpoint')
        if not os.path.exists(path_checkpoint):
            os.makedirs(path_checkpoint)
        save_path = os.path.join(path_checkpoint, "checkpoint_epoch[%d_%d].pth" % (epoch, cfg.total_epoch))
        torch.save(save, save_path)
        self.logger.info("Save checkpoint to {}".format(save_path))

    def _make_batch_loader(self):
        dataset_path = "./data/DatasetPKL"
        if os.path.isfile(os.path.join(dataset_path, cfg.dataset, 'train_dataloader.pkl')) and os.path.isfile(os.path.join(dataset_path, cfg.dataset, 'valid_dataloader.pkl')):
            self.logger.info("Loading dataloader...")
            self.train_loader = torch.load(os.path.join(dataset_path, cfg.dataset, 'train_dataloader.pkl'))
            self.logger.info("The dataset is loaded successfully.")
        else:
            self.logger.info("Creating dataset...")
            dataset = get_dataset(cfg.dataset, cfg.dataset_path ,'training')
            split = int(len(dataset) * 0.8)
            train_split, valid_split = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
            self.train_loader = DataLoader(train_split,                                       
                                        batch_size=cfg.batch_size,
                                        num_workers=cfg.num_worker,
                                        shuffle=True,
                                        pin_memory=True,
                                        drop_last=True)

            self.valid_loader = DataLoader(valid_split,
                                        batch_size=cfg.batch_size,
                                        num_workers=cfg.num_worker,
                                        shuffle=False,
                                        pin_memory=True)
            if not os.path.isdir(os.path.join(dataset_path, cfg.dataset)):
                os.makedirs(os.path.join(dataset_path, cfg.dataset))
            torch.save(self.train_loader, os.path.join(dataset_path, cfg.dataset, 'train_dataloader.pkl'))
            torch.save(self.valid_loader, os.path.join(dataset_path, cfg.dataset, 'valid_dataloader.pkl'))
            self.logger.info("The dataset is created successfully.")
        
    def _make_model(self):
        self.logger.info("Making the model...")
        model = get_model().to(cfg.device)
        optimizer = self.get_optimizer(model)
        schedule = self.get_schedule(optimizer)
        if cfg.continue_train:
            start_epoch, model = self.load_model(model, optimizer, schedule)
        else:
            start_epoch = 0
        
        if cfg.use_multigpu:
            self.model = nn.DataParallel(self.model)
            self.logger.info("Training of Multiple GPUs")
        model.train()
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.logger.info("The model is made successfully.")

class Tester:
    def __init__(self):
        cfg.output_root = './valid/%s'%cfg.dataset
        today = datetime.strftime(datetime.now().replace(microsecond=0), '%Y-%m-%d')
        base_folder = os.path.join(cfg.output_root, "%s_%d"%(today, cfg.trial_num))
        if os.path.exists(base_folder):
            cfg.trial_num += 1
            base_folder = os.path.join(cfg.output_root, "%s_%d"%(today, cfg.trial_num))
            os.makedirs(base_folder)
            cfg.base_folder = base_folder

        log_folder = os.path.join(cfg.base_folder, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'eval_' + cfg.experiment_name + '.log')
        with open(os.path.join(log_folder, 'train_' + cfg.experiment_name + '.yaml')) as fw:
            yaml.safe_dump(vars(cfg), fw, default_flow_style=False)

        self.logger = setup_logger(output=logfile, name="%s_valid"%cfg.pre)
        self.logger.info('Start evaluation: %s' % ('eval_' + cfg.experiment_name))

    def _make_batch_loader(self):
        dataset_path = "./data/DatasetPKL"
        if os.path.isfile(os.path.join(dataset_path, cfg.dataset, 'train_dataloader.pkl')) and os.path.isfile(os.path.join(dataset_path, cfg.dataset, 'valid_dataloader.pkl')):
            self.logger.info("Loading dataloader...")
            self.valid_loader = torch.load(os.path.join(dataset_path, cfg.dataset, 'valid_dataloader.pkl'))
            self.logger.info("The dataset is loaded successfully.")
        else:
            self.logger.error("No valid dataset loader")

    def load_model(self, model):
        self.logger.info('Loading the model from {}...'.format(cfg.checkpoint))
        checkpoint = torch.load(cfg.checkpoint)
        model.load_state_dict(checkpoint['net'])
        self.logger.info('The model is loaded successfully.')
        return model

    def _make_model(self):
        self.logger.info("Making the model...")
        model = get_model().to(cfg.device)
        model = self.load_model(model)
        model.eval()
        self.model = model
        self.logger.info("The model is made successfully.")

    def _evaluate(self, outs, meta_info, cur_sample_idx):
        eval_result = self.dataset.evaluate(outs, meta_info, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.dataset.print_eval_result(eval_result)
        self.logger.info("The evaluation is done successfully.")