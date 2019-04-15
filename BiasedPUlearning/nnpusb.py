import numpy as np
from scipy import optimize
from model import MultiLayerPerceptron, CNN
from train import train

import chainer
from chainer import cuda, Function, gradient_check, Variable
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class nnPU:
    def __init__(self, pi):
        self.pi = pi
        self.loss_func = lambda g: self.loss(g)

    def loss(self, g):
        g = np.log(1+np.exp(-g))
        return g
    
    def pu(self, x, b, t, reg):
        xp = x[t == 1]
        xu = x[t == 0]
        n1 = len(xp)
        n0 = len(xu)
        gp = np.dot(xp, b)
        gu = np.dot(xu, b)
        loss_u = self.loss_func(-gu)
        J1 = -(self.pi/n1)*np.sum(gp)
        J0 = (1/n0)*np.sum(loss_u)
        J = J1+J0+reg*np.dot(b,b)
        return J
    
    def prob(self, x, b):
        x = self.x
        g = np.dot(x, b)
        prob = 1/(1+np.exp(-g))
        return prob


    def test(self, x, b, t, quant=True, pi=False):
        theta = 0
        f = np.dot(x, b)
        if quant is True:
            temp = np.copy(f)
            temp = np.sort(temp)
            theta = temp[np.int(np.floor(len(x)*(1-pi)))]
        pred = np.zeros(len(x))
        pred[f > theta] = 1
        acc = np.mean(pred == t)
        return acc

    def dist(self, x, T=None, num_basis=False):

        (d,n) = x.shape

        # check input argument

        # set the kernel bases
        X = x

        if num_basis is False:
            num_basis = n

        idx = np.random.permutation(n)[0:num_basis]
        C = X[:, idx]

        # calculate the squared distances
        XC_dist = CalcDistanceSquared(x, C)
        TC_dist = CalcDistanceSquared(T, C)
        CC_dist = CalcDistanceSquared(C, C)

        return XC_dist, TC_dist, CC_dist, n, num_basis

    def kernel_cv(self, x_train, t, x_test, folds=5, num_basis=False, sigma_list=None, lda_list=None):
        x_train, x_test = x_train.T, x_test.T
        XC_dist, TC_dist, CC_dist, n, num_basis = self.dist(x_train, x_test, num_basis)
        # setup the cross validation
        cv_fold = np.arange(folds) # normal range behaves strange with == sign
        cv_split0 = np.floor(np.arange(n)*folds/n)
        cv_index = cv_split0[np.random.permutation(n)]
        # set the sigma list and lambda list
        if sigma_list==None:
            sigma_list = np.array([0.5, 1, 2, 5, 10])
        if lda_list==None:
            lda_list = np.array([0.001, 0.01, 0.1, 1.])

        score_cv = np.zeros((len(sigma_list), len(lda_list)))

        for sigma_idx, sigma in enumerate(sigma_list):

            # pre-sum to speed up calculation
            h_cv = []
            t_cv = []
            for k in cv_fold:
                h_cv.append(np.exp(-XC_dist[:, cv_index==k]/(2*sigma**2)))
                t_cv.append(t[cv_index==k])

            for k in range(folds):
                #print(h0_cv[0])
                # calculate the h vectors for training and test
                count = 0
                for j in range(folds):
                    if j == k:
                        hte = h_cv[j].T
                        tte = t_cv[j]
                    else:
                        if count == 0:
                            htr = h_cv[j].T
                            ttr = t_cv[j]
                            count += 1
                        else:
                            htr = np.append(htr, h_cv[j].T, axis=0)
                            ttr = np.append(ttr, t_cv[j], axis=0)

                one = np.ones((len(htr),1))
                htr = np.concatenate([htr, one], axis=1)
                one = np.ones((len(hte),1))
                hte = np.concatenate([hte, one], axis=1)
                for lda_idx, lda in enumerate(lda_list):
                    res = self.minimize(htr, ttr, lda)
                    # calculate the solution and cross-validation value
    
                    score = self.pu(hte, res, tte, lda)
                    """
                    if math.isnan(score):
                        code.interact(local=dict(globals(), **locals()))
                    """
                    
                    score_cv[sigma_idx, lda_idx] = score_cv[sigma_idx, lda_idx] + score

        # get the minimum
        (sigma_idx_chosen, lda_idx_chosen) = np.unravel_index(np.argmin(score_cv), score_cv.shape)
        sigma_chosen = sigma_list[sigma_idx_chosen]
        lda_chosen = lda_list[lda_idx_chosen]

        #print(score_cv)
        #print(sigma_chosen)
        #print(lda_chosen)

        # calculate the new solution
        x_train = np.exp(-XC_dist/(2*sigma_chosen**2)).T
        x_test = np.exp(-TC_dist/(2*sigma_chosen**2)).T

        return x_train, x_test, lda_chosen

    def linear_cv(x0, x1, folds=5, lda_list=None):
        if lda_list==None:
            lda_list = np.array([0.001, 0.01, 0.1, 1.])

        scores = []
        for lda in lda_list:
            func = lambda b: self.linear_uu(b, x0, x1)
            res = self.minimize(func, b0)
            score = func(res)
            scores.append(score)

        scores = np.array(scores)
        lda = lda_list[scores.argmin()]
        return lda

def CalcDistanceSquared(X, C):
    '''
    Calculates the squared distance between X and C.
    XC_dist2 = CalcDistSquared(X, C)
    [XC_dist2]_{ij} = ||X[:, j] - C[:, i]||2
    :param X: dxn: First set of vectors
    :param C: d:nc Second set of vectors
    :return: XC_dist2: The squared distance nc x n
    '''

    Xsum = np.sum(X**2, axis=0).T
    Csum = np.sum(C**2, axis=0)
    XC_dist = Xsum[np.newaxis, :] + Csum[:, np.newaxis] - 2*np.dot(C.T, X)
    return XC_dist

        
    
