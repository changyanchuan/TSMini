import os
import random
import torch
import numpy

def set_seed(seed = -1):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    
    debug = False
    dumpfile_uniqueid = ''
    seed = 2000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.abspath(__file__)[:-10] # dont use os.getcwd()
    checkpoint_dir = root_dir + '/exp/snapshots'

    dataset = 'porto'
    dataset_prefix = ''
    dataset_file = ''

    #===========general=============
    min_traj_len = 20
    max_traj_len = 200
    cell_size = 100.0
    cellspace_buffer = 500.0 
    
    traj_duplicate_short_tolerance = 80
    traj_distance_norm_denominator = 1000


    #===========TSMini=============
    seq_embedding_dim = 128

    tsmini_conv_channel_dim = 64
    tsmini_conv_hidden_in_dim = 16
    tsmini_conv_kernel_size = 3
    tsmini_conv_stride = 1
    tsmini_patch_emb_dim = 128 # same to seq_embedding_dim
    
    tsmin_trans_attention_head = 4
    tsmin_trans_attention_dropout = 0.1
    tsmin_trans_attention_layer = 1
    tsmin_trans_pos_encoder_dropout = 0.1
    tsmin_trans_hidden_dim = 512


    #===========trajsimi=============
    trajsimi_encoder_name = 'TSMini'
    trajsimi_measure_fn_name = 'frechet'
    trajsimi_loss_mse_weight = 0.2
    
    trajsimi_batch_size = 128 # 128
    trajsimi_epoch = 40
    trajsimi_training_bad_patience = 5
    trajsimi_learning_rate = 0.002 # 0.0001 # 0.0002
    trajsimi_training_lr_degrade_step = 15 # 5
    trajsimi_training_lr_degrade_gamma = 0.5
    trajsimi_learning_weight_decay = 0.00001
    trajsimi_finetune_lr_rescale = 0.5


    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()


    @classmethod
    def post_value_updates(cls):
        if 'porto' == cls.dataset:
            cls.dataset_prefix = 'porto_20200'
        elif 'xian' == cls.dataset:
            cls.dataset_prefix = 'xian_20200'
        else:
            pass
        
        cls.dataset_file = cls.root_dir + '/data/' + cls.dataset_prefix
        set_seed(cls.seed)


    @classmethod
    def to_str(cls): # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
                        lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                        dic.items() \
                        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
