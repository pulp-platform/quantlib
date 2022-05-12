from .preprocess import ILSVRC12MNv2Transform
from .mobilenetv2 import MobileNetV2
from .headrewriter import MNv2HeadRewriter

import os
from typing import List, Union


def MNv2_checkpoints() -> List[Union[os.PathLike, str]]:

    path_package = os.path.dirname(os.path.realpath(__file__))
    path_checkpoints = os.path.join(path_package, 'checkpoints')

    if not os.path.isdir(path_checkpoints):
        raise FileNotFoundError

    checkpoints = [os.path.join(path_checkpoints, filename) for filename in os.listdir(path_checkpoints) if (filename.endswith('.ckpt'))]

    return checkpoints
