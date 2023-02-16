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
        self.backbone = 'resnet34'
        self.num_stage = 2
        self.num_FMs = 8
        self.feature_size = 64
        self.heatmap_size = 32
        self.num_vert = 778
        self.num_joint = 21
        # training
        self.batch_size = 64
        self.lr = 3e-4
        self.total_epoch = 50
        self.input_img_shape = (256, 256)
        self.depth_box = 0.3
        self.num_worker = 16
        self.loss_queries = {}
        self.loss_weight = {}
        # -------------
        self.save_epoch = 1
        self.eval_interval = 1
        self.print_iter = 10
        self.print_epoch = 100
        self.num_epoch_to_eval = 80
        # -------------
        self.checkpoint = './model/SAR-R34-S2-65-67.pth'  # put the path of the trained model's weights here
        self.continue_train = False
        self.vis = True
    # -------------
        self.experiment_name = self.pre + '_{}'.format(self.dataset) + '_Batch{}'.format(self.batch_size) + '_lr{}'.format(self.lr) + '_Epochs{}'.format(self.total_epoch)

cfg = Config()
