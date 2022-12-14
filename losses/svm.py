import torch
import torch.nn as nn
import numpy as np
import losses.functional as F

from losses.utils import detect_large


def SmoothSVM(n_classes, alpha=None, tau=1., k=5, loss_weight=None):
    if k == 1:
        return SmoothTop1SVM(n_classes, alpha, tau)
    else:
        return SmoothTopkSVM(n_classes, alpha, tau, k, loss_weight)


class _SVMLoss(nn.Module):

    def __init__(self, n_classes, alpha, loss_weight):

        assert isinstance(n_classes, int)

        assert n_classes > 0
        assert alpha is None or alpha >= 0

        super(_SVMLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 1
        self.register_buffer('labels', torch.Tensor(np.arange(n_classes)))
        self.n_classes = n_classes
        self._tau = None
        self.loss_weight = loss_weight

    def forward(self, x, y):

        raise NotImplementedError("Forward needs to be re-implemented for each loss")

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self._tau != tau:
            print("Setting tau to {}".format(tau))
            self._tau = float(tau)
            self.get_losses()

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        self.get_losses()

    def cpu(self):
        nn.Module.cpu()
        self.get_losses()


class MaxTop1SVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None):

        super(MaxTop1SVM, self).__init__(n_classes=n_classes,
                                         alpha=alpha)
        self.get_losses()

    def forward(self, x, y):
        return self.F(x, y).mean()

    def get_losses(self):
        self.F = F.Top1_Hard_SVM(self.labels, self.alpha)


class MaxTopkSVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None, k=5):

        super(MaxTopkSVM, self).__init__(n_classes=n_classes,
                                         alpha=alpha)
        self.k = k
        self.get_losses()

    def forward(self, x, y):
        return self.F(x, y).mean()

    def get_losses(self):
        self.F = F.Topk_Hard_SVM(self.labels, self.k, self.alpha)


class SmoothTop1SVM(_SVMLoss):
    def __init__(self, n_classes, alpha=None, tau=1.):
        super(SmoothTop1SVM, self).__init__(n_classes=n_classes,
                                            alpha=alpha)
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(self, x, y):
        smooth, hard = detect_large(x, 1, self.tau, self.thresh)

        loss = 0
        if smooth.data.sum():
            x_s, y_s = x[smooth[:, None]], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            loss += self.F_s(x_s, y_s).sum() / x.size(0)
        if hard.data.sum():
            x_h, y_h = x[hard[:, None]], y[hard]
            x_h = x_h.view(-1, x.size(1))
            loss += self.F_h(x_h, y_h).sum() / x.size(0)

        return loss

    def get_losses(self):
        self.F_h = F.Top1_Hard_SVM(self.labels, self.alpha)
        self.F_s = F.Top1_Smooth_SVM(self.labels, self.tau, self.alpha)


class SmoothTopkSVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None, tau=1., k=5, loss_weight=None):
        super(SmoothTopkSVM, self).__init__(n_classes=n_classes,
                                            alpha=alpha, loss_weight=loss_weight)
        self.k = k
        self.tau = tau
        self.thresh = 1e3
        self.loss_weight = loss_weight
        self.get_losses()

    def forward(self, x, y):
        smooth, hard = detect_large(x, self.k, self.tau, self.thresh)

        loss = 0
        if smooth.data.sum():
            # x_s, y_s = x[smooth[:, None]], y[smooth]
            x_s, y_s = x[smooth], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            loss += self.F_s(x_s, y_s).sum() / x.size(0)
            # print('----------- smooth!!!!!!!!!!!!!!!')
        if hard.data.sum():
            # x_h, y_h = x[hard[:, None]], y[hard]
            x_h, y_h = x[hard], y[hard]
            x_h = x_h.view(-1, x.size(1))
            loss += self.F_h(x_h, y_h).sum() / x.size(0)
            # print('----------- hard!!!!!!!!!!!!!!!')

        return loss

    def get_losses(self):
        self.F_h = F.Topk_Hard_SVM(self.labels, self.k, self.alpha, self.loss_weight)
        self.F_s = F.Topk_Smooth_SVM(self.labels, self.k, self.tau, self.alpha, self.loss_weight)
