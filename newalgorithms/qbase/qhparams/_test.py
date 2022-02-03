import time
import torch

from quantlib.newalgorithms.qbase.qrange import IMPLICIT_STEP
from quantlib.newalgorithms.qbase.qhparams.qclipper import get_scale


def main(n_channels: int = 512, n_levels: int = 256):

    n_levels = torch.ones((n_channels,)) * n_levels

    times = []
    for _ in range(0, 1000):

        # generate random interval and random zero-point (i.e., offset)
        a = torch.randn(n_channels)
        b = a + torch.rand(n_channels)
        z = torch.randint(low=-256, high=255, size=(len(a),)).to(dtype=torch.float32)
        s = torch.ones(size=(len(a),)) * IMPLICIT_STEP

        # compute scale (i.e., quantum) while profiling
        st = time.time()
        eps = get_scale(a, b, z, n_levels, s)
        en = time.time()

        times.append(en - st)

    print(torch.Tensor(times).mean())


if __name__ == '__main__':
    main()
