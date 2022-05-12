import os
import torch
import torchvision

from .transforms import ILSVRC12STATS


def get_ilsvrc12_dataset(transform: torchvision.transforms.Compose) -> torch.utils.data.Dataset:

    path_package = os.path.dirname(os.path.realpath(__file__))
    path_data = os.path.join(path_package, 'data', 'val')
    
    if not os.path.isdir(path_data):
        raise FileNotFoundError

    dataset = torchvision.datasets.ImageFolder(path_data, transform)
    dataset.scale = torch.Tensor([ILSVRC12STATS['quantize']['eps']])

    return dataset
