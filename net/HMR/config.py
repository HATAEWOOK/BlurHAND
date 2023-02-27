import torch

class Config(object):
    def __init__(self):
        self.manual_seed = 77
        self.trial_num = 2
        self.pre = 'HMR_SV'
        self.dataset = 'FreiHAND'
        self.dataset_path = '/mnt/workplace/dataset/freiHand_dataset'
        self.split = 0.05
        self.device = "cuda"
        self.use_multigpu = True
        # network
        self.lr = 1e-3
        self.batch_size = 16
        self.num_worker = 20
        self.pretrained = True
        # training
        self.total_epoch = 50
        # -------------
        self.save_epoch = 1
        self.eval_interval = 1
        self.print_iter = 10
        self.print_epoch = 100
        self.num_epoch_to_eval = 80
        # -------------
        self.loss_queries = ['j3d_hm', 'j3d_vol', 'j2d_hm']
        self.loss_weight = {'j3d_vol' : 1e1, 
            'j3d_hm' : 1e1,
            'j2d_hm' : 1, }
        # self.loss_queries = ['j2d_hm']
        # self.loss_weight = {'j2d_hm' : 1, }
        # -------------
        self.pretrained_net = "/mnt/workplace/blurHand/models/pretrained/S24_100_net.pt"
        self.checkpoint = "/mnt/workplace/blurHand/out/train/FreiHAND/2023-02-17_0/checkpoint/checkpoint_epoch[99_100].pth"  # put the path of the trained model's weights here
        self.continue_train = True
        self.vis = True
    # -------------
        self.experiment_name = self.pre + '_{}'.format(self.dataset) + '_Batch{}'.format(self.batch_size) + '_lr{}'.format(self.lr) + '_Epochs{}'.format(self.total_epoch)

cfg = Config()
