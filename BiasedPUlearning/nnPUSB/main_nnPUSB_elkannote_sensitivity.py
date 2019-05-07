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
    ite = 10
    pdata = 1000
    epoch = 100
    batchsize = 1000

    seed = 2018

    gpu = True

    loss_pu = np.zeros((ite, epoch))
    est_error_pubp0 = np.zeros((ite, epoch))
    est_error_pubp1 = np.zeros((ite, epoch))
    est_error_pubp2 = np.zeros((ite, epoch))
    est_error_pubp3 = np.zeros((ite, epoch))
    est_pre-acc_pu0 = np.zeros((ite, epoch))
    est_pre-acc_pu1 = np.zeros((ite, epoch))
    est_pre-acc_pu2 = np.zeros((ite, epoch))
    est_pre-acc_pu3 = np.zeros((ite, epoch))
    
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

        model, optimizer, loss_list, acc, acc0, pre, rec, pre0, rec0 = train_pu(x_train, t_train, x_test, t_test, pi-0.05, epoch, model, optimizer, batchsize, xp)
        model, optimizer, loss_list, acc, acc1, pre, rec, pre1, rec1 = train_pu(x_train, t_train, x_test, t_test, pi+0.05, epoch, model, optimizer, batchsize, xp)        
        model, optimizer, loss_list, acc, acc2, pre, rec, pre2, rec2 = train_pu(x_train, t_train, x_test, t_test, pi+0.1, epoch, model, optimizer, batchsize, xp)        
        model, optimizer, loss_list, acc, acc3, pre, rec, pre3, rec3 = train_pu(x_train, t_train, x_test, t_test, pi+0.2, epoch, model, optimizer, batchsize, xp)

        loss_pu[i] = loss_list
        est_error_pubp0[i] = acc0
        est_error_pubp1[i] = acc1
        est_error_pubp2[i] = acc2
        est_error_pubp3[i] = acc3
        est_pre_acc_senstive0[i] = pre0 - acc0 
        est_pre_acc_senstive1[i] = pre1 - acc1
        est_pre_acc_senstive2[i] = pre2 - acc2
        est_pre_acc_senstive3[i] = pre3 - acc3

        est_precision_pu[i] = pre1
        est_recall_pu[i] = rec1
        est_precision_pubp[i] = pre2
        est_recall_pubp[i] = rec2

        seed += 1
        
        np.savetxt('loss_pu_elkan_%d.csv'%seed, loss_pu, delimiter=',')
        np.savetxt('est_error_pubp_elkan_senstive(-0.05)_%d.csv'%seed, est_error_pubp0, delimiter=',')
        np.savetxt('est_error_pubp_elkan_senstive(+0.05)_%d.csv'%seed, est_error_pubp1, delimiter=',')
        np.savetxt('est_error_pubp_elkan_senstive(+0.1)_%d.csv'%seed, est_error_pubp2, delimiter=',')
        np.savetxt('est_error_pubp_elkan_senstive(+0.2)_%d.csv'%seed, est_error_pubp3, delimiter=',')
        np.savetxt('est_pre-acc_senstive(-0.05)_%d.csv'%seed, est_pre_acc_senstive0, delimiter=',')
        np.savetxt('est_pre-acc_senstive(+0.05)_%d.csv'%seed, est_pre_acc_senstive1, delimiter=',')
        np.savetxt('est_pre-acc_senstive(+0.1)_%d.csv'%seed, est_pre_acc_senstive2, delimiter=',')
        np.savetxt('est_pre-acc_senstive(+0.2)_%d.csv'%seed, est_pre_acc_senstive3, delimiter=',')

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
