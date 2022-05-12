from ..data import ILSVRC12Transform


class ILSVRC12RNTransform(ILSVRC12Transform):

    def __init__(self):
        rn_image_size = 224
        super(ILSVRC12RNTransform, self).__init__(image_size=rn_image_size)
