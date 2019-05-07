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

    xp = np.loadtxt('xp.csv')
    xq = np.loadtxt('xq.csv')
    xn = np.loadtxt('xn.csv')

    x_train = np.concatenate([xp, xq, xn], axis=0)
    t_train = np.concatenate([np.ones(len(xp)), np.zeros(len(xq)+len(xn))], axis=0)
    x_test = np.concatenate([xq, xn], axis=0)
    t_test = np.concatenate([np.ones(len(xq)), np.zeros(len(xn))], axis=0)
    for i in range(ite):
        np.random.seed(seed)
        
        print(x_train.shape)
        print(t_train.shape)
        print(x_test.shape)
        print(t_test.shape)

        pi = np.mean(t_test)

        print(pi)

        dim = x_train.shape[1]        

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

        model, optimizer, loss_list, acc1, acc2, pre1, rec1, pre2, rec2 = train_pu(x_train, t_train, x_test, t_test, pi, epoch, model, optimizer, batchsize, xp)


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
        
        np.savetxt('result/est_error_pu_elkan.csv', est_error_pu, delimiter=',')
        np.savetxt('result/est_error_pubp_elkan.csv', est_error_pubp, delimiter=',')
        np.savetxt('result/est_precision_pu_elkan.csv', est_precision_pu, delimiter=',')
        np.savetxt('result/est_recall_pu_elkan.csv', est_recall_pu, delimiter=',')
        np.savetxt('result/est_precision_pubp_elkan.csv', est_precision_pubp, delimiter=',')
        np.savetxt('result/est_recall_pubp_elkan.csv', est_recall_pubp, delimiter=',')

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
