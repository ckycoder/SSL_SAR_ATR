import torch

from future.builtins import range
from losses.logarithm import LogTensor


def recursion(S, X, j):
    """
    Apply recursive formula to compute the gradient
    for coefficient of degree j.
    d S[j] / d X = S[j-1] - X * (S[j-2] - X * (S[j-3] - ...) ... )
                 = S[j-1] + X ** 2 * S[j-3] + ...
                 - (X * S[j-2] + X ** 3 * S[j-4] + ...)
    """

    """ original code """
    _P_ = sum(S[i] * X ** (j - 1 - i) for i in range(j - 1, -1, -2))
    _N_ = sum(S[i] * X ** (j - 1 - i) for i in range(j - 2, -1, -2))
    return _N_, _P_


    """ modified by @Alien """
    # Compute positive and negative parts separately
    # if j == 1:
    #     _P_ = sum(S[i] * X ** (j - 1 - i) for i in range(j - 1, -1, -2))
    #     return _P_
    # else:
    #     _P_ = sum(S[i] * X ** (j - 1 - i) for i in range(j - 1, -1, -2))
    #     _N_ = sum(S[i] * X ** (j - 1 - i) for i in range(j - 2, -1, -2))
    #     return _N_, _P_
    """ modified by @Alien """


def approximation(S, X, j, p):
    """
    Compute p-th order approximation for d S[j] / d X:
    d S[j] / d X ~ S[j] / X - S[j + 1] /  X ** 2 + ...
                   + (-1) ** (p - 1) * S[j + p - 1] / X ** p
    """

    # Compute positive and negative parts separately
    _P_ = sum(S[j + i] / X ** (i + 1) for i in range(0, p, 2))
    _N_ = sum(S[j + i] / X ** (i + 1) for i in range(1, p, 2))

    return _N_, _P_


def d_logS_d_expX(S, X, j, p, grad, thresh, eps=1e-5):
    """
    Compute the gradient of log S[j] w.r.t. exp(X).
    For unstable cases, use p-th order approximnation.
    """

    # ------------------------------------------------------------------------
    # Detect unstabilites
    # ------------------------------------------------------------------------

    _X_ = LogTensor(X)
    _S_ = [LogTensor(S[i]) for i in range(S.size(0))]

    """ original code """
    # recursion of gradient formula (separate terms for stability)
    _N_, _P_ = recursion(_S_, _X_, j)

    # detect instability: small relative difference in log-space
    P, N = _P_.torch(), _N_.torch()
    diff = (P - N) / (N.abs() + eps)

    """ modified by @Alien """
    # recursion of gradient formula (separate terms for stability)
    # if j == 1:
    #     _P_ = recursion(_S_, _X_, j)
    #     P = _P_.torch()
    #     diff = P / eps
    # else:
    #     _N_, _P_ = recursion(_S_, _X_, j)
    #
    #     # detect instability: small relative difference in log-space
    #     P, N = _P_.torch(), _N_.torch()
    #     diff = (P - N) / (N.abs() + eps)
    """ modified by @Alien """

    # split into stable and unstable indices
    u_indices = torch.lt(diff, thresh)  # unstable
    s_indices = u_indices.eq(0)  # stable

    # ------------------------------------------------------------------------
    # Compute d S[j] / d X
    # ------------------------------------------------------------------------

    # make grad match size and type of X
    grad = grad.type_as(X).resize_as_(X)

    """ original code """
    # exact gradient for s_indices (stable) elements
    if s_indices.sum():
        # re-use positive and negative parts of recursion (separate for stability)
        _N_ = LogTensor(_N_.torch()[s_indices])
        _P_ = LogTensor(_P_.torch()[s_indices])
        _X_ = LogTensor(X[s_indices])
        _S_ = [LogTensor(S[i][s_indices]) for i in range(S.size(0))]

        # d log S[j] / d exp(X) = (d S[j] / d X) * X / S[j]
        _SG_ = (_P_ - _N_) * _X_ / _S_[j]
        grad.masked_scatter_(s_indices, _SG_.torch().exp())

    """ modified by @Alien """
    # exact gradient for s_indices (stable) elements
    # if s_indices.sum():
    #     # re-use positive and negative parts of recursion (separate for stability)
    #     if j == 1:
    #         # _N_ = LogTensor(_N_.torch()[s_indices])
    #         _P_ = LogTensor(_P_.torch()[s_indices])
    #         _X_ = LogTensor(X[s_indices])
    #         _S_ = [LogTensor(S[i][s_indices]) for i in range(S.size(0))]
    #
    #         _SG_ = _P_ * _X_ / _S_[j]
    #     else:
    #         _N_ = LogTensor(_N_.torch()[s_indices])
    #         _P_ = LogTensor(_P_.torch()[s_indices])
    #         _X_ = LogTensor(X[s_indices])
    #         _S_ = [LogTensor(S[i][s_indices]) for i in range(S.size(0))]
    #
    #         # d log S[j] / d exp(X) = (d S[j] / d X) * X / S[j]
    #         _SG_ = (_P_ - _N_) * _X_ / _S_[j]
    #
    #     grad.masked_scatter_(s_indices, _SG_.torch().exp())
    """ modified by @Alien """

    # approximate gradients for u_indices (unstable) elements
    if u_indices.sum():
        _X_ = LogTensor(X[u_indices])
        _S_ = [LogTensor(S[i][u_indices]) for i in range(S.size(0))]

        # positive and negative parts of approximation (separate for stability)
        _N_, _P_ = approximation(_S_, _X_, j, p)

        # d log S[j] / d exp(X) = (d S[j] / d X) * X / S[j]
        _UG_ = (_P_ - _N_) * _X_ / _S_[j]
        grad.masked_scatter_(u_indices, _UG_.torch().exp())


    """ modified by @Alien, 18.11.20, 11:51 """
    # print('--- grad ---')
    # print(grad)

    return grad
