import numpy as np
import pandas as pd
import six
import matplotlib.pyplot as plt

from dataset_nnPU import *
from pubp_nn import PUNN
from model import *
from train_nnPU import * 

import chainer.cuda
import chainer

def experiment():
    ite = 100
    pdata = 1000
    epoch = 100
    batchsize = 1000

    seed = 2018

    gpu = True

    loss_pu = np.zeros((ite, epoch))
    est_error_pu = np.zeros((ite, epoch))
    est_error_pubp = np.zeros((ite, epoch))
    est_precision_pu = np.zeros((ite, epoch))
    est_recall_pu = np.zeros((ite, epoch))
    est_precision_pubp = np.zeros((ite, epoch))
    est_recall_pubp = np.zeros((ite, epoch))

    for i in range(ite):
        np.random.seed(seed)
        #PN classification
        x_train, t_train, x_test, t_test = load_dataset("mnist")
        t_train[t_train == -1] = 0
        t_test[t_test == -1] = 0

        pi = np.mean(t_train)

        x = np.concatenate([x_train, x_test], axis=0)
        t = np.concatenate([t_train, t_test], axis=0)
        x = x.reshape(x.shape[0], x.shape[2]*x.shape[3])
        dim = x.shape[1]
        print(x.shape)

        model = MultiLayerPerceptron(dim)
        optimizer = optimizers.Adam(1e-5)
        optimizer.setup(model)

        if gpu:
            gpu_device = 0
            cuda.get_device(gpu_device).use()
            model.to_gpu(gpu_device)
            xp = cuda.cupy
        else:
            xp = np

        model, optimizer = train(x, t, epoch, model, optimizer, batchsize, xp)

        x_p = x_train[t_train==1]

        xp_prob = np.array([])
        for j in six.moves.range(0, len(x_p), batchsize):
            X = Variable(xp.array(x_p[j:j + batchsize], xp.float32))
            g = chainer.cuda.to_cpu(model(X).data).T[0]
            xp_prob = np.append(xp_prob, 1/(1+np.exp(-g)), axis=0)
        xp_prob /= np.mean(xp_prob)
        xp_prob = xp_prob
        xp_prob /= np.max(xp_prob)
        print(xp_prob)
        rand = np.random.uniform(size=len(x_p))
        x_p = x_p[xp_prob > rand]
        perm = np.random.permutation(len(x_p))
        x_p = x_p[perm[:pdata]]

        tp = np.ones(len(x_p))
        tu = np.zeros(len(x_train))
        t_train = np.concatenate([tp, tu], axis=0)

        x_train = np.concatenate([x_p, x_train], axis=0)

        print(x_train.shape)
        print(t_train.shape)
        print(x_test.shape)
        print(t_test.shape)

        model = MultiLayerPerceptron(dim)
        optimizer = optimizers.Adam(alpha=1e-5)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.005))

        if gpu:
            gpu_device = 0
            cuda.get_device(gpu_device).use()
            model.to_gpu(gpu_device)
            xp = cuda.cupy
        else:
            xp = np

        model, optimizer, loss_list, acc1, acc2, pre1, rec1, pre2, rec2  = train_pu(x_train, t_train, x_test, t_test, pi, epoch, model, optimizer, batchsize, xp)

        loss_pu[i] = loss_list
        est_error_pu[i] = acc1
        est_error_pubp[i] = acc2
        est_precision_pu[i] = pre1
        est_recall_pu[i] = rec1
        est_precision_pubp[i] = pre2
        est_recall_pubp[i] = rec2

        print(acc1[-1])
        print(acc2[-1])

        seed += 1

        np.savetxt('loss_pu_mnist_%d.csv'%seed, loss_pu, delimiter=',')
        np.savetxt('est_error_pu_mnist_%d.csv'%seed, est_error_pu, delimiter=',')
        np.savetxt('est_error_pubp_mnist__%d.csv'%seed, est_error_pubp, delimiter=',')
        np.savetxt('est_precision_pu_mnist_%d.csv'%seed, est_precision_pu, delimiter=',')
        np.savetxt('est_recall_pu_mnist_%d.csv'%seed, est_recall_pu, delimiter=',')
        np.savetxt('est_precision_pubp_mnist_%d.csv'%seed, est_precision_pubp, delimiter=',')
        np.savetxt('est_recall_pubp_mnist_%d.csv'%seed, est_recall_pubp, delimiter=',')

    loss_pu_mean = np.mean(loss_pu, axis=1)
    est_error_pu_mean = np.mean(est_error_pu, axis=1)
    est_error_pubp_mean = np.mean(est_error_pubp, axis=1)
    est_error_pu_std = np.std(est_error_pu, axis=1)
    est_error_pubp_std = np.std(est_error_pubp, axis=1)
    return loss_pu_mean, est_error_pu_mean, est_error_pubp_mean, est_error_pu_std, est_error_pubp_std 

def main():
    loss_pu_mean, est_error_pu_mean, est_error_pubp_mean, est_error_pu_std, est_error_pubp_std = experiment()
    print(loss_pu_mean)
    print(est_error_pu_mean)
    print(est_error_pubp_mean)

if __name__ == "__main__":
    main()
