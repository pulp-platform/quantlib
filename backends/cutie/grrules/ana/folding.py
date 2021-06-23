import numpy as np
import torch


def fold_anaact_anaconv2d_bn2d_anaact(eps_x: torch.Tensor,
                                      eps_w: torch.Tensor, weight: torch.Tensor,
                                      mi: torch.Tensor, sigma: torch.Tensor, bn_eps: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                                      eps_s: torch.Tensor, theta: torch.Tensor, ceiltau: bool = True):

    def torch2numpyfp64(x):
        return x.detach().cpu().numpy().astype(np.float64)

    eps_x  = torch2numpyfp64(eps_x)
    eps_w  = torch2numpyfp64(eps_w)
    weight = torch2numpyfp64(weight)
    mi     = torch2numpyfp64(mi)
    sigma  = torch2numpyfp64(sigma)
    gamma  = torch2numpyfp64(gamma)
    beta   = torch2numpyfp64(beta)
    eps_s  = torch2numpyfp64(eps_s)
    theta  = torch2numpyfp64(theta)

    # compensate for negative gammas
    flip   = np.sign(gamma)
    w_tmp  = weight.transpose(1, 2, 3, 0)
    w_tmp *= flip
    weight = w_tmp.transpose(3, 0, 1, 2)

    # https://github.com/pytorch/pytorch/blob/b5e832111e5e4bb3dd66d716d398b81fe70c6af0/torch/csrc/jit/tensorexpr/kernel.cpp#L2015
    sigma = np.sqrt(sigma + bn_eps)

    # folding
    xi   = gamma * (sigma ** -1)
    zeta = beta - mi * xi

    gammaprime = flip * (xi * (eps_x * eps_w) / eps_s)
    betaprime  = zeta / eps_s

    # prepare for broadcasting
    gammaprime = np.expand_dims(gammaprime, axis=0)
    betaprime  = np.expand_dims(betaprime, axis=0)
    theta      = np.expand_dims(theta, axis=-1)

    # absorb folded parameters into thresholds
    tau = (theta - betaprime) / gammaprime

    assert np.all(tau[0] < tau[1])

    def numpy2torchfp64(x):
        return torch.from_numpy(x.astype(np.float64))

    if ceiltau:
        return numpy2torchfp64(np.ceil(tau)).float(), numpy2torchfp64(weight).float()
    else:
        return numpy2torchfp64(tau).float(), numpy2torchfp64(weight).float()


def fold_anaact_analinear_bn1d_anaact(eps_x: torch.Tensor,
                                      eps_w: torch.Tensor, weight: torch.Tensor,
                                      mi: torch.Tensor, sigma: torch.Tensor, bn_eps: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                                      eps_s: torch.Tensor, theta: torch.Tensor):

    def torch2numpyfp64(x):
        return x.detach().cpu().numpy().astype(np.float64)

    eps_x  = torch2numpyfp64(eps_x)
    eps_w  = torch2numpyfp64(eps_w)
    weight = torch2numpyfp64(weight)
    mi     = torch2numpyfp64(mi)
    sigma  = torch2numpyfp64(sigma)
    gamma  = torch2numpyfp64(gamma)
    beta   = torch2numpyfp64(beta)
    eps_s  = torch2numpyfp64(eps_s)
    theta  = torch2numpyfp64(theta)

    # compensate for negative gammas
    flip   = np.sign(gamma)
    w_tmp  = weight.transpose(1, 0)
    w_tmp *= flip
    weight = w_tmp.transpose(1, 0)

    # https://github.com/pytorch/pytorch/blob/b5e832111e5e4bb3dd66d716d398b81fe70c6af0/torch/csrc/jit/tensorexpr/kernel.cpp#L2015
    sigma = np.sqrt(sigma + bn_eps)

    # folding
    xi   = gamma * (sigma ** -1)
    zeta = beta - mi * xi

    gammaprime = flip * (xi * (eps_x * eps_w) / eps_s)
    betaprime  = zeta / eps_s

    # prepare for broadcasting
    gammaprime = np.expand_dims(gammaprime, axis=0)
    betaprime  = np.expand_dims(betaprime, axis=0)
    theta      = np.expand_dims(theta, axis=-1)

    # absorb folded parameters into thresholds
    tau = (theta - betaprime) / gammaprime

    assert np.all(tau[0] < tau[1])

    def numpy2torchfp64(x):
        return torch.from_numpy(x.astype(np.float64))

    return numpy2torchfp64(np.ceil(tau)).float(), numpy2torchfp64(weight).float()
