#Source : https://github.com/zxz267/SAR 
import torch
import torch.nn as nn
import yaml
import os
from torch.utils.data import DataLoader
from datetime import datetime
import torch.optim as optim
# from net.HMR.config import cfg
# from net.SAR.config import cfg
from net.evalNet.config import cfg
from utils.logger import setup_logger
from data.dataset import get_dataset
from net.model import get_model
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self):
        self.device = torch.device(cfg.device)
        # cfg.output_root = '/mnt/workplace/blurHand/out/train/%s'%cfg.dataset
        cfg.output_root = '/root/workplace/backup/blurHand/out/train/%s'%cfg.dataset
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
            if cfg.continue_train:
                optimizer.load_state_dict(checkpoint['optimizer'])
                schedule.load_state_dict(checkpoint['schedule'])
                start_epoch = checkpoint['last_epoch'] + 1
                cfg.total_epoch += checkpoint['last_epoch']
            self.logger.info('Model loaded')
        return start_epoch, model

    def load_pretrained(self, model, mode='front'):
        if cfg.pretrained_net is not None:
            state = torch.load(cfg.pretrained_net)
            volumenet_decoder_state = {k[len("vt_net.volume_net.encoder_decoder."):]:v for k, v in state.items() if 'volume_net' in k and 'encoder_decoder' in k and '.decoder_' in k}
            volumenet_backlayers_state = {k[len("vt_net.volume_net."):]:v for k, v in state.items() if 'volume_net' in k and 'back_layers' in k}
            volumenet_outputlayer_state = {k[len("vt_net.volume_net."):]:v for k, v in state.items() if 'volume_net' in k and 'output_layer' in k}
            volumenet_backlayers_state.update(volumenet_outputlayer_state)
            model.volume_net.encoder_decoder.load_state_dict(volumenet_decoder_state, strict=False)
            model.volume_net.load_state_dict(volumenet_backlayers_state, strict=False)
            if cfg.freeze:
                for name, param in model.named_parameters():
                    if 'volume_net' in name and 'encoder_decoder' in name and '.decoder_' in name:
                        param.requires_grad = False
                        print(name, param.requires_grad)
                    if 'volume_net' in name and ('back_layers' in name or 'output_layer' in name):
                        param.requires_grad = False
                        print(name, param.requires_grad)
            if mode == 'back':
                state = torch.load(cfg.pretrained_net)
                regressor_state = {k[len("regressor."):]:v for k, v in state.items() if 'regressor' in k and 'mano' not in k}
                model.regressor.load_state_dict(regressor_state)
                if cfg.freeze:
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
        if isinstance(cfg.split, float) and os.path.isfile(os.path.join(dataset_path, cfg.dataset, 'train_dataloader_%.2f.pkl'%cfg.split)):
            self.logger.info("Loading dataloader...")
            self.train_loader = torch.load(os.path.join(dataset_path, cfg.dataset, 'train_dataloader_%.2f.pkl'%cfg.split))
            self.logger.info("The dataset is loaded successfully.")
        elif isinstance(cfg.split, int) and os.path.isfile(os.path.join(dataset_path, cfg.dataset, 'train_dataloader_%d.pkl'%cfg.split)):
            self.logger.info("Loading dataloader...")
            self.train_loader = torch.load(os.path.join(dataset_path, cfg.dataset, 'train_dataloader_%d.pkl'%cfg.split))
            self.logger.info("The dataset is loaded successfully.")
        elif cfg.split is None and os.path.isfile(os.path.join(dataset_path, cfg.dataset, 'train_dataloader.pkl')):
            self.logger.info("Loading dataloader...")
            self.train_loader = torch.load(os.path.join(dataset_path, cfg.dataset, 'train_dataloader.pkl'))
            self.logger.info("The dataset is loaded successfully.")
        else:
            self.logger.info("Creating dataset...")
            dataset = get_dataset(cfg.dataset, cfg.dataset_path ,'training')
            if isinstance(cfg.split, float): #split by ratio
                split = int(len(dataset) * cfg.split)
                train_loader, last_split = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
                valid_loader, last_valid_split = torch.utils.data.random_split(last_split, [int(split/4), len(last_split) - int(split/4)])
                del last_split, last_valid_split
            elif isinstance(cfg.split, int): #split by length
                split = cfg.split
                train_loader, last_split = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
                valid_loader, last_valid_split = torch.utils.data.random_split(last_split, [int(split/4), len(last_split) - int(split/4)])
                del last_split, last_valid_split
            else:
                split = int(len(dataset)*0.8)
                train_loader, valid_loader = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

            self.train_loader = DataLoader(train_loader,                                       
                                        batch_size=cfg.batch_size,
                                        num_workers=cfg.num_worker,
                                        shuffle=True,
                                        pin_memory=True,
                                        drop_last=True)

            self.valid_loader = DataLoader(valid_loader,
                                        batch_size=cfg.batch_size,
                                        num_workers=cfg.num_worker,
                                        shuffle=False,)
            
            if not os.path.isdir(os.path.join(dataset_path, cfg.dataset)):
                os.makedirs(os.path.join(dataset_path, cfg.dataset))

            if isinstance(split, float):
                torch.save(self.train_loader, os.path.join(dataset_path, cfg.dataset, 'train_dataloader_%.2f.pkl'%split))
                torch.save(self.valid_loader, os.path.join(dataset_path, cfg.dataset, 'valid_dataloader_%.02f.pkl'%split/4))
            elif isinstance(split, int):
                torch.save(self.train_loader, os.path.join(dataset_path, cfg.dataset, 'train_dataloader_%d.pkl'%split))
                torch.save(self.valid_loader, os.path.join(dataset_path, cfg.dataset, 'valid_dataloader_%d.pkl'%(int(split/4))))
            elif split is None:
                torch.save(self.train_loader, os.path.join(dataset_path, cfg.dataset, 'train_dataloader.pkl'))
                torch.save(self.valid_loader, os.path.join(dataset_path, cfg.dataset, 'valid_dataloader.pkl'))
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
            model = self.load_pretrained(model, mode='back')
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
        self.device = torch.device(cfg.device)
        cfg.output_root = '/mnt/workplace/blurHand/out/valid/%s'%cfg.dataset
        today = datetime.strftime(datetime.now().replace(microsecond=0), '%Y-%m-%d')
        cfg.base_folder = os.path.join(cfg.output_root, "%s_%d"%(today, cfg.trial_num))
        if os.path.exists(cfg.base_folder):
            raise "Log Directory exist"

        log_folder = os.path.join(cfg.base_folder, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'eval_' + cfg.experiment_name + '.log')
        with open(os.path.join(log_folder, 'eval_' + cfg.experiment_name + '.yaml'), "w") as fw:
            yaml.safe_dump(vars(cfg), fw, default_flow_style=False)

        self.logger = setup_logger(output=logfile, name="%s_valid"%cfg.pre)
        self.logger.info('Start evaluation: %s' % ('eval_' + cfg.experiment_name))

    def _make_batch_loader(self):
        dataset_path = "./data/DatasetPKL"
        if os.path.isfile(os.path.join(dataset_path, cfg.dataset, 'valid_dataloader.pkl')):
            self.logger.info("Loading dataloader...")
            self.valid_loader = torch.load(os.path.join(dataset_path, cfg.dataset, 'valid_dataloader.pkl'))
            self.logger.info("The dataset is loaded successfully.")
        else:
            self.logger.error("No valid dataset loader")

    def load_model(self, model):
        self.logger.info('Loading the model from {}...'.format(cfg.checkpoint))
        checkpoint = torch.load(cfg.checkpoint)
        state = {k[len("module."):]:v for k, v in checkpoint['net'].items()}
        model.load_state_dict(state, strict=False)
        self.logger.info('The model is loaded successfully.')
        return model
    
    def load_pretrained(self, model, mode='front'):
        if cfg.pretrained_net is not None:
            state = torch.load(cfg.pretrained_net)
            volumenet_decoder_state = {k[len("vt_net.volume_net.encoder_decoder."):]:v for k, v in state.items() if 'volume_net' in k and 'encoder_decoder' in k and '.decoder_' in k}
            volumenet_backlayers_state = {k[len("vt_net.volume_net."):]:v for k, v in state.items() if 'volume_net' in k and 'back_layers' in k}
            volumenet_outputlayer_state = {k[len("vt_net.volume_net."):]:v for k, v in state.items() if 'volume_net' in k and 'output_layer' in k}
            volumenet_backlayers_state.update(volumenet_outputlayer_state)
            model.volume_net.encoder_decoder.load_state_dict(volumenet_decoder_state, strict=False)
            model.volume_net.load_state_dict(volumenet_backlayers_state, strict=False)
            if mode == 'back':
                regressor_state = {k[len("regressor."):]:v for k, v in state.items() if 'regressor' in k and 'mano' not in k}
                model.regressor.load_state_dict(regressor_state)
        return model

    def _make_model(self):
        self.logger.info("Making the model...")
        model = get_model(cfg.pre)
        model = self.load_model(model)
        if cfg.pretrained_net is not None:
            model = self.load_pretrained(model, mode='back')
        model.eval()
        self.model = model.cpu()
        self.logger.info("The model is made successfully.")

    def visualization(self, out, iter, image):
        vis_path = os.path.join(cfg.base_folder, 'vis')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        fig = plt.figure()
        plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
        plt.imshow(out['re_img'][0].cpu().numpy())
        fig.savefig(os.path.join(vis_path, '%s_vis.png'%iter))