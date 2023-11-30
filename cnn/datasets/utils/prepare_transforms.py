from torchvision import transforms


def prepare_transforms(cfg):
    return transforms.Compose([getattr(transforms, name)(*params) for name, params in cfg])