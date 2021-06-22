__all__ = ['Controller']


class Controller(object):

    def __init__(self):
        # controller base class does not do any initialization
        pass

    @staticmethod
    def get_modules(*args, **kwargs):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step_pre_training_epoch(self, *args, **kwargs):
        """Update the quantization hyper-parameters before the training step of the current epoch."""
        raise NotImplementedError

    def step_pre_training_batch(self, *args, **kwargs):
        """Update the quantization hyper-parameters before the training mini-batch."""
        pass

    def step_pre_validation_epoch(self, *args, **kwargs):
        """Update the quantization hyper-parameters before the validation step of the current epoch."""
        raise NotImplementedError
