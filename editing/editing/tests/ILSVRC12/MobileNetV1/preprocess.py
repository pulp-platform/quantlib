from ..data import ILSVRC12Transform


class ILSVRC12MNv1Transform(ILSVRC12Transform):

    def __init__(self):
        mnv1_image_size = 224
        super(ILSVRC12MNv1Transform, self).__init__(image_size=mnv1_image_size)
