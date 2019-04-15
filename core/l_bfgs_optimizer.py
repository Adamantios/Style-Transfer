from scipy.optimize import fmin_l_bfgs_b
import numpy as np


class LBFGSOptimizer(object):
    """
    This class makes it possible to compute loss and gradients in one pass
    while retrieving them via two separate functions, "_loss" and "_grads".
    This is done because scipy.optimize requires separate functions for loss and gradients,
    but computing them separately would be inefficient.
    """

    def __init__(self, img_nrows, img_ncols, f_outputs):
        self.img_nrows = img_nrows
        self.img_ncols = img_ncols
        self.f_outputs = f_outputs
        self.loss = None
        self.grads = None

    def _eval_loss_and_grads(self, x):
        x = x.reshape((1, self.img_nrows, self.img_ncols, 3))

        outputs = self.f_outputs([x])
        loss = outputs[0]
        if len(outputs[1:]) == 1:
            grads = outputs[1].flatten().astype('float64')
        else:
            grads = np.array(outputs[1:]).flatten().astype('float64')
        return loss, grads

    def _loss(self, x):
        loss_value, grad_values = self._eval_loss_and_grads(x)
        self.loss = loss_value
        self.grads = grad_values
        return self.loss

    def _grads(self, _):
        grad_values = np.copy(self.grads)
        self.loss = None
        self.grads = None
        return grad_values

    def run(self, x0, n_iter: int):
        # run scipy-based optimization (L-BFGS) over the pixels of the generated image
        # so as to minimize the neural style loss
        x, f, _ = fmin_l_bfgs_b(self._loss, x0, self._grads, maxfun=n_iter)
        return x, f
