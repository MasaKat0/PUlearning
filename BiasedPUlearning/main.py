import numpy as np
import pandas as pd
import six

from dataset_for_nnPU import *
from model import *
from train import * 

import chainer.cuda
import chainer

def experiment():
    ite = 100
    pdata = 100
    epoch = 100
    batchsize = 1000

    seed = 2018

    gpu = False

    loss_pu = np.zeros((ite, epoch))
    est_error_pu = np.zeros((ite, epoch))
    est_error_pusb = np.zeros((ite, epoch))

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

        model = ThreeLayerPerceptron(dim)
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
        xp_prob = xp_prob**10
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

        model = ThreeLayerPerceptron(dim)
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

        model, optimizer, loss_list, acc0, acc1 = train_pu(x_train, t_train, x_test, t_test, pi, epoch, model, optimizer, batchsize, xp)
        #precison0, precison1, recall0, recall1

        loss_pu[i] = loss_list
        est_error_pu[i] = acc0
        est_error_pusb[i] = acc1

        print(acc0[-1])
        print(acc1[-1])

        seed += 1
        
        np.savetxt('result/loss_pu_mnist.csv', loss_pu, delimiter=',')
        np.savetxt('result/error_pu_mnist.csv', est_error_pu, delimiter=',')
        np.savetxt('result/error_pubp_mnist.csv', est_error_pusb, delimiter=',')
        np.savetxt('result/recall_pu_mnist.csv', est_recall_pu, delimiter=',')
        np.savetxt('result/precision_pu_mnist.csv', est_precision_pu, delimiter=',')
        np.savetxt('result/est_recall_pusb_mnist.csv', est_recall_pusb, delimiter=',')
        np.savetxt('result/est_precision_pusb_mnist.csv', est_precision_pusb, delimiter=',')


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
