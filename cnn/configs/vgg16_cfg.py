from easydict import EasyDict

cfg = EasyDict()

cfg.conv_blocks = {"in_channels": [[3, 64], [64, 128], [128, 256, 256], [256, 512, 512], [512, 512, 512]],
                   "out_channels": [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]}

cfg.full_conn_blocks = {"in_features": [512*7*7, 4096],
                        "out_features": [4096, 4096]}

cfg.momentum = 0.9

