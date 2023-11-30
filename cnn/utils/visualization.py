import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms import transforms


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    plt.figure(figsize=(18, 18))
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_batch(batch_images, nrow=8, inv_normalize=True):
    plt.cla(), plt.clf()
    if inv_normalize:
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        batch_images = inv_normalize(batch_images)

    grid = torchvision.utils.make_grid(batch_images, nrow=nrow)
    show(grid)
    plt.show()
    return plt.gcf()
