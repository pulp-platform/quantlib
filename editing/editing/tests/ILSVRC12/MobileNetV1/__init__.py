from .preprocess import ILSVRC12MNv1Transform
from .mobilenetv1 import MobileNetV1
from .headrewriter import MNv1HeadRewriter

import os
from typing import List, Union


def MNv1_checkpoints() -> List[Union[os.PathLike, str]]:

    path_package = os.path.dirname(os.path.realpath(__file__))
    path_checkpoints = os.path.join(path_package, 'checkpoints')

    if not os.path.isdir(path_checkpoints):
        raise FileNotFoundError

    checkpoints = [os.path.join(path_checkpoints, filename) for filename in os.listdir(path_checkpoints) if (filename.endswith('.ckpt'))]

    return checkpoints
