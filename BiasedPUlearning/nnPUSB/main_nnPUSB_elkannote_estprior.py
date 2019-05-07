import numpy as np
import pandas as pd
import six
import matplotlib.pyplot as plt

from dataset_nnPU  *
from pubp_nn import PUNN
from model import *
from train_nnPU import * 
from ramaswamy import *

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
    est_precision1_pubp = np.zeros((ite, epoch))
    est_recall1_pubp = np.zeros((ite, epoch))
    est_precision2_pubp = np.zeros((ite, epoch))
    est_recall2_pubp = np.zeros((ite, epoch))
    est_precision3_pubp = np.zeros((ite, epoch))
    est_recall3_pubp = np.zeros((ite, epoch))
    est_precision4_pubp = np.zeros((ite, epoch))
    est_recall4_pubp = np.zeros((ite, epoch))
    est_precision5_pubp = np.zeros((ite, epoch))
    est_recall5_pubp = np.zeros((ite, epoch))

    xp = np.loadtxt('xp.csv')
    xq = np.loadtxt('xq.csv')
    xn = np.loadtxt('xn.csv')

    x_train = np.concatenate([xp, xq, xn], axis=0)
    t_train = np.concatenate([np.ones(len(xp)), np.zeros(len(xq)+len(xn))], axis=0)
    x_test = np.concatenate([xq, xn], axis=0)
    t_test = np.concatenate([np.ones(len(xq)), np.zeros(len(xn))], axis=0)
    for i in range(ite):
        np.random.seed(seed)
        #PN classification
        
        (KM1, KM2) = wrapper(x_train[t_train==0],x_train[t_train==1])
        pi = KM2
        #pi = np.mean(t_test)

        print(x_train.shape)
        print(t_train.shape)
        print(x_test.shape)
        print(t_test.shape)
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

        model, optimizer, loss_list, acc1, acc2, pre0, rec0, pre1, rec1, pre2, rec2, pre3, rec3, pre4, rec4, pre5, rec5, pre, rec = train_pu(x_train, t_train, x_test, t_test, pi, epoch, model, optimizer, batchsize, xp)
       
        if i == 0:
            pres = pre
            recs = rec
        else:
            pres += pre
            recs += rec

        loss_pu[i] = loss_list
        est_error_pu[i] = acc1
        est_error_pubp[i] = acc2
        est_precision_pu[i] = pre0
        est_recall_pu[i] = rec0
        est_precision1_pubp[i] = pre1
        est_recall1_pubp[i] = rec1
        est_precision2_pubp[i] = pre2
        est_recall2_pubp[i] = rec2
        est_precision3_pubp[i] = pre3
        est_recall3_pubp[i] = rec3
        est_precision4_pubp[i] = pre4
        est_recall4_pubp[i] = rec4
        est_precision5_pubp[i] = pre5
        est_recall5_pubp[i] = rec5


        print(acc1[-1])
        print(acc2[-1])

        seed += 1
        
        np.savetxt('loss_pu_elkan_%d.csv'%seed, loss_pu, delimiter=',')
        np.savetxt('est_error_pu_elkan_%d.csv'%seed, est_error_pu, delimiter=',')
        np.savetxt('est_error_pubp_elkan_%d.csv'%seed, est_error_pubp, delimiter=',')
        np.savetxt('est_precision_pu_elkan_%d.csv'%seed, est_precision_pu, delimiter=',')
        np.savetxt('est_recal-l_pu_elkan_%d.csv'%seed, est_recall_pu, delimiter=',')
        np.savetxt('est_precision-1_pubp_elkan_%d.csv'%seed, est_precision1_pubp, delimiter=',')
        np.savetxt('est_recall-1_pubp_elkan_%d.csv'%seed, est_recall1_pubp, delimiter=',')
        np.savetxt('est_precision-2_pubp_elkan_%d.csv'%seed, est_precision2_pubp, delimiter=',')
        np.savetxt('est_recall-2_pubp_elkan_%d.csv'%seed, est_recall2_pubp, delimiter=',')
        np.savetxt('est_precision-3_pubp_elkan_%d.csv'%seed, est_precision3_pubp, delimiter=',')
        np.savetxt('est_recall-3_pubp_elkan_%d.csv'%seed, est_recall3_pubp, delimiter=',')
        np.savetxt('est_precision-4_pubp_elkan_%d.csv'%seed, est_precision4_pubp, delimiter=',')
        np.savetxt('est_recall-4_pubp_elkan_%d.csv'%seed, est_recall4_pubp, delimiter=',')
        np.savetxt('est_precision-5_pubp_elkan_%d.csv'%seed, est_precision5_pubp, delimiter=',')
        np.savetxt('est_recall-5_pubp_elkan_%d.csv'%seed, est_recall5_pubp, delimiter=',')
        np.savetxt('precisions_pubp_elkan_%d.csv'%seed, pres/(i+1), delimiter=',')
        np.savetxt('recalls_pubp_elkan_%d.csv'%seed, recs/(i+1), delimiter=',')
        
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
