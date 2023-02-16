import torch

class Config(object):
    def __init__(self):
        self.manual_seed = 77
        self.trial_num = 0
        self.pre = 'SAR'
        self.dataset = 'FreiHAND'
        self.dataset_path = '/root/workplace/ssd/dataset/freiHand_dataset'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_multigpu = True
        # network
        self.lr = 1e-3
        self.batch_size = 8
        self.pretrained = True
        # training
        self.total_epoch = 200
        # -------------
        self.save_epoch = 1
        self.eval_interval = 1
        self.print_iter = 10
        self.print_epoch = 100
        self.num_epoch_to_eval = 80
        # -------------
        self.pretrained_net = None
        self.checkpoint = None  # put the path of the trained model's weights here
        self.continue_train = False
        self.vis = True
    # -------------
        self.experiment_name = self.pre + '_{}'.format(self.dataset) + '_Batch{}'.format(self.batch_size) + '_lr{}'.format(self.lr) + '_Epochs{}'.format(self.total_epoch)

cfg = Config()
