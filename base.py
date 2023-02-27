#Source : https://github.com/zxz267/SAR 
import torch
import torch.nn as nn
import yaml
import os
from torch.utils.data import DataLoader
from datetime import datetime
import torch.optim as optim
from net.HMR.config import cfg
from utils.logger import setup_logger
from data.dataset import get_dataset
from net.model import get_model

class Trainer:
    def __init__(self):
        self.device = torch.device(cfg.device)
        cfg.output_root = '/mnt/workplace/blurHand/out/train/%s'%cfg.dataset
        today = datetime.strftime(datetime.now().replace(microsecond=0), '%Y-%m-%d')
        cfg.base_folder = os.path.join(cfg.output_root, "%s_%d"%(today, cfg.trial_num))
        if os.path.exists(cfg.base_folder):
            raise "Log Directory exist"
        
        log_folder = os.path.join(cfg.base_folder, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'train_' + cfg.experiment_name + '.log')
        with open(os.path.join(log_folder, 'train_' + cfg.experiment_name + '.yaml'), 'w') as w:
            yaml.safe_dump(vars(cfg), w, default_flow_style=False)

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
            state = {k[len("module."):]:v for k, v in checkpoint['net'].items()}
            model.load_state_dict(state)
            optimizer.load_state_dict(checkpoint['optimizer'])
            schedule.load_state_dict(checkpoint['schedule'])
            start_epoch = checkpoint['last_epoch'] + 1
            cfg.total_epoch += checkpoint['last_epoch']
            self.logger.info('Model loaded')
        return start_epoch, model

    def load_pretrained(self, model, mode='front'):
        if cfg.pretrained_net is not None:
            if mode == 'front':
                state = torch.load(cfg.pretrained_net)
                volumenet_decoder_state = {k[len("vt_net.volume_net.encoder_decoder."):]:v for k, v in state.items() if 'volume_net' in k and 'encoder_decoder' in k and '.decoder_' in k}
                volumenet_backlayers_state = {k[len("vt_net.volume_net."):]:v for k, v in state.items() if 'volume_net' in k and 'back_layers' in k}
                volumenet_outputlayer_state = {k[len("vt_net.volume_net."):]:v for k, v in state.items() if 'volume_net' in k and 'output_layer' in k}
                volumenet_backlayers_state.update(volumenet_outputlayer_state)
                model.volume_net.encoder_decoder.load_state_dict(volumenet_decoder_state, strict=False)
                model.volume_net.load_state_dict(volumenet_backlayers_state, strict=False)
                for name, param in model.named_parameters():
                    if 'volume_net' in name and 'encoder_decoder' in name and '.decoder_' in name:
                        param.requires_grad = False
                        print(name, param.requires_grad)
                    if 'volume_net' in name and ('back_layers' in name or 'output_layer' in name):
                        param.requires_grad = False
                        print(name, param.requires_grad)
            elif mode == 'back':
                state = torch.load(cfg.pretrained_net)
                regressor_state = {k[len("vt_net.regressor."):]:v for k, v in state.items() if 'regressor' in k}
                model.regressor.load_state_dict(regressor_state)
                for name, param in model.named_parameters():
                    if 'regressor' not in name:
                        param.requires_grad = False
                        print(name, param.requires_grad)



        return model

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
        save_path = os.path.join(path_checkpoint, "checkpoint_epoch[%d_%d].pth" % (epoch+1, cfg.total_epoch))
        torch.save(save, save_path)
        self.logger.info("Save checkpoint to {}".format(save_path))

    def _make_batch_loader(self):
        dataset_path = "./data/DatasetPKL"
        # if os.path.isfile(os.path.join(dataset_path, cfg.dataset, 'train_dataloader_%.2f.pkl'%cfg.split)):
        if os.path.isfile(os.path.join(dataset_path, cfg.dataset, 'train_dataloader.pkl')):
            self.logger.info("Loading dataloader...")
            self.train_loader = torch.load(os.path.join(dataset_path, cfg.dataset, 'train_dataloader.pkl'))
            self.logger.info("The dataset is loaded successfully.")
        else:
            self.logger.info("Creating dataset...")
            dataset = get_dataset(cfg.dataset, cfg.dataset_path ,'training')
            split = int(len(dataset) * cfg.split)
            train_split, valid_split = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
            del valid_split
            self.train_loader = DataLoader(train_split,                                       
                                        batch_size=cfg.batch_size,
                                        num_workers=cfg.num_worker,
                                        shuffle=True,
                                        pin_memory=True,
                                        drop_last=True)

            # self.valid_loader = DataLoader(valid_split,
            #                             batch_size=cfg.batch_size,
            #                             num_workers=cfg.num_worker,
            #                             shuffle=False,)
            if not os.path.isdir(os.path.join(dataset_path, cfg.dataset)):
                os.makedirs(os.path.join(dataset_path, cfg.dataset))
            torch.save(self.train_loader, os.path.join(dataset_path, cfg.dataset, 'train_dataloader_%.2f.pkl'%cfg.split))
            # torch.save(self.valid_loader, os.path.join(dataset_path, cfg.dataset, 'valid_dataloader_%.02f.pkl'%(1-cfg.split)))
            self.logger.info("The dataset is created successfully.")
        
    def _make_model(self):
        self.logger.info("Making the model...")
        model = get_model(cfg.pre).to(self.device)
        optimizer = self.get_optimizer(model)
        schedule = self.get_schedule(optimizer)
        if cfg.continue_train:
            start_epoch, model = self.load_model(model, optimizer, schedule)
        else:
            start_epoch = 1
        if cfg.pretrained_net:
            model = self.load_pretrained(model)
        model.train()
        self.model = model
        if cfg.use_multigpu:
            self.model = nn.DataParallel(self.model, output_device=1)
            self.logger.info("Training of Multiple GPUs")
        self.start_epoch = start_epoch
        self.optimizer = optimizer
        self.schedule = schedule
        self.logger.info("The model is made successfully.")

class Tester:
    def __init__(self):
        cfg.output_root = './valid/%s'%cfg.dataset
        today = datetime.strftime(datetime.now().replace(microsecond=0), '%Y-%m-%d')
        cfg.base_folder = os.path.join(cfg.output_root, "%s_%d"%(today, cfg.trial_num))
        if os.path.exists(cfg.base_folder):
            raise "Log Directory exist"

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
        model = get_model().to(self.device)
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