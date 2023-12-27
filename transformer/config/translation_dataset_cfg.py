from easydict import EasyDict

cfg = EasyDict()
cfg.max_sentence_len = 512
cfg.languages = ['en', 'ru']
cfg.voc_size_en = 30000
cfg.voc_size_ru = 30000
