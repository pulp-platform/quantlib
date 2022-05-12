from ..data import ILSVRC12Transform


class ILSVRC12MNv2Transform(ILSVRC12Transform):

    def __init__(self):
        mnv2_image_size = 224
        super(ILSVRC12MNv2Transform, self).__init__(image_size=mnv2_image_size)
