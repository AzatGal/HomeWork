from easydict import EasyDict

cfg = EasyDict()

cfg.dmodel = 512
cfg.h = 8
cfg.N = 6


cfg.dropout = 0.1
cfg.batch_size = 32
cfg.device = 'cpu'
cfg.epoch = 20

cfg.b1 = 0.9
cfg.b2 = 0.98
cfg.eps_opt = 1e-9

cfg.num_epoch = 15

cfg.max_search_len = 100
