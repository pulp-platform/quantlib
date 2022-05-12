from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import Compose


ILSVRC12STATS = \
    {
        'normalize':
            {
                'mean': (0.485, 0.456, 0.406),
                'std':  (0.229, 0.224, 0.225)
            },
        'quantize':
            {
                'min': -2.1179039478,
                'max': 2.6400001049,
                'eps': 0.020625000819563866
            }
    }


class ILSVRC12Normalize(Normalize):
    def __init__(self):
        super(ILSVRC12Normalize, self).__init__(**ILSVRC12STATS['normalize'])


class ILSVRC12Transform(Compose):

    def __init__(self, image_size: int = 224):

        # validate arguments
        RESIZE_SIZE = 256
        if not (image_size <= RESIZE_SIZE):
            raise ValueError  # otherwise, `CenterCrop` can not yield the desired image

        transforms = [Resize(RESIZE_SIZE),
                      CenterCrop(image_size),
                      ToTensor(),
                      ILSVRC12Normalize()]

        super(ILSVRC12Transform, self).__init__(transforms)
