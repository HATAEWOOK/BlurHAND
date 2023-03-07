import torch

class Config(object):
    def __init__(self):
        self.manual_seed = 77
        self.trial_num = 0
        self.pre = 'evalNet'
        # self.dataset = 'FreiHAND'
        # self.dataset_path = '/root/workplace/ssd/dataset/freiHand_dataset'
        self.dataset = 'blurHand'
        self.dataset_path = '/root/workplace/ssd/dataset/blurhand_new'
        # self.split = 7420
        self.split = None
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
        self.loss_queries = ['j3d', 'j2d', 'mask', 'reg', 'beta']
        self.loss_weight = {'j3d' : 1e2, 
                            'j2d' : 1, 
                            'mask':1e2, 
                            'reg':1e1, 
                            'beta':1e5}
        # -------------
        self.pretrained_net = None
        self.checkpoint = None  # put the path of the trained model's weights here
        self.continue_train = False
        self.vis = False
    # -------------
        self.experiment_name = self.pre + '_{}'.format(self.dataset) + '_Batch{}'.format(self.batch_size) + '_lr{}'.format(self.lr) + '_Epochs{}'.format(self.total_epoch)

cfg = Config()
