import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, cuda, function, Variable
import numpy as np

class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(None, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 1)
        )
        self.af = F.relu

    def calculate(self, x):
        h = self.l1(x)
        h = self.af(h)
        h = self.l2(h)
        h = self.af(h)
        h = self.l3(h)
        return h

class PULoss(function.Function):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: F.log(F.sigmoid(-x)))):
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.loss_func = loss
        self.positive = 1
        self.unlabeled = 0

    def forward(self, x_in, x, t):
        x_in = x_in
        positive, unlabeled = t == self.positive, t == self.unlabeled
        n_positive, n_unlabeled = np.sum(positive), np.sum(unlabeled)
        y_positive = self.loss_func(x_in)
        y_unlabeled = self.loss_func(-x_in)
        risk1 = self.prior * F.sum(x * positive, axis=0) / n_positive
        risk2 = F.sum(F.batch_matmul(x, F.sigmoid(-x_in)), axis=0) / n_unlabeled
        self.loss = - risk1 + risk2.T[0]
        print(np.sum(risk1.data))
        print(np.sum(risk2.T[0].data))
        print(F.sum(self.loss).data)
        return self.loss
