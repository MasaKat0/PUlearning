import numpy as np
import six
from scipy import optimize

import chainer
from chainer import cuda, Function, gradient_check, Variable
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

def train(x_train, t_train, epoch, model, optimizer, batchsize=5000, xp=np):
    N = len(x_train)
    t_train[t_train == 0] = -1
    for ep in six.moves.range(1, epoch + 1):
        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N, batchsize):
            model.zerograds()

            x = Variable(xp.array(x_train[perm[i:i + batchsize]], xp.float32))
            t = Variable(xp.array([t_train[perm[i:i + batchsize]]], xp.float32).T)

            # Pass the loss function (Classifier defines it) and its arguments
            g = model(x)
            loss = F.mean(F.log(1+F.exp(-t*g)))
            loss.backward()
            optimizer.update()

    return model, optimizer

def train_pu(x_train, t_train, x_test, t_test, pi, epoch, model, optimizer, batchsize=5000, xp=np):
    N = len(x_train)
    loss_list = []
    acc1_list = []
    acc2_list = []
    for ep in six.moves.range(1, epoch + 1):
        # training
        perm = np.random.permutation(N)
        loss_step = 0
        count = 0
        for i in six.moves.range(0, N, batchsize):
            model.zerograds()

            x = Variable(xp.array(x_train[perm[i:i + batchsize]], xp.float32))
            t_temp = t_train[perm[i:i + batchsize]]
            t_temp = xp.array([t_temp], xp.float32).T
            t = Variable(t_temp)

            #np = xp.sum(t)
            #nu = batchsize - n1
            # Pass the loss function (Classifier defines it) and its arguments
            g = model(x)
            positive, unlabeled = t_temp == 1, t_temp == 0
            n_p = max([1, xp.sum(positive)])
            n_u = max([1, xp.sum(unlabeled)])
            gp = F.log(1+F.exp(-g))
            gu = F.log(1+F.exp(g))
            lossp = pi*F.sum(gp*positive)/n_p
            lossn = F.sum(gu*unlabeled)/n_u - pi*F.sum(gu*positive)/n_p
            if lossn.data < 0:
                loss = -lossn
            else:
                loss = lossp + lossn
            
            loss.backward()
            optimizer.update()
            loss_step += loss.data
            count += 1
        loss_step /= count
        loss_list.append(loss_step)
        acc1 = test(x_test, t_test, model, quant=False, xp=xp, batchsize=batchsize)
        acc2 = test(x_test, t_test, model, quant=True, pi=pi, xp=xp, batchsize=batchsize)
        acc1_list.append(acc1)
        acc2_list.append(acc2)

    loss_list = np.array(loss_list)
    acc1_list = np.array(acc1_list)
    acc2_list = np.array(acc2_list)

    return model, optimizer, loss_list, acc1_list, acc2_list

def test(x, t, model, quant=True, pi=False, xp=np, batchsize=100):
    theta = 0
    f = np.array([])
    for i in six.moves.range(0, len(x), batchsize):
        X = Variable(xp.array(x[i:i + batchsize], xp.float32))
        p = chainer.cuda.to_cpu(model(X).data).T[0]
        f = np.append(f, p, axis=0)
    if quant is True:
        temp = np.copy(f)
        temp = np.sort(temp)
        theta = temp[np.int(np.floor(len(x)*(1-pi)))]
    pred = np.zeros(len(x))
    pred[f > theta] = 1
    acc = np.mean(pred == t)
    return acc