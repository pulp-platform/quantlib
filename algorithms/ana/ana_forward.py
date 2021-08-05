import torch


def forward_expectation(pmf: torch.Tensor, q: torch.Tensor):
    q_shape = [q.numel()] + [1 for _ in range(1, pmf.dim())]  # prepare for broadcasting
    return torch.sum(pmf * q.reshape(q_shape), dim=0)


def forward_mode(pmf: torch.Tensor, q: torch.Tensor):
    return q[torch.argmax(pmf, dim=0)]


def forward_random(pmf: torch.Tensor, q: torch.Tensor):
    q_shape = [q.numel()] + [1 for _ in range(1, pmf.dim())]  # prepare for broadcasting
    us = torch.rand_like(pmf[0]).unsqueeze(0)
    ge = torch.cumsum(pmf, dim=0) - us  # compute the position of the random numbers with respect to the segments
    sc = (ge[:-1] * ge[1:] <= 0).float()  # detect sign change
    idxs = torch.vstack([1.0 - torch.sum(sc, dim=0), sc])
    return torch.sum(idxs * q.reshape(q_shape), dim=0)
