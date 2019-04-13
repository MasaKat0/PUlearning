import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from dataset2 import make_data2
from dataset3 import make_data3
from pubp_nn import PUNN
from model import *
from train import * 
from model_dr import *
from train_dr import *

def experiment():
    ite = 100
    pdata = 10000
    epoch = 100
    batchsize = 30000

    seed = 2018
    
    gpu = False

    loss_pu = np.zeros((ite, epoch))
    est_error_pu = np.zeros((ite, epoch))
    est_error_pubp = np.zeros((ite, epoch))
    est_error_dr = np.zeros((ite, epoch))

    for i in range(ite):
        np.random.seed(seed)
        #PN classification
        x, t = make_data3(datatype="digits")
        pi = np.mean(t)
        dim = x.shape[1]

        model = normalNN(dim)
        optimizer = optimizers.Adam()
        optimizer.setup(model)

        if gpu:
            gpu_device = 0
            cuda.get_device(gpu_device).use()
            model.to_gpu(gpu_device)
            xp = cuda.cupy
        else:
            xp = np

        model, optimizer = train(x, t, epoch, model, optimizer, batchsize, xp)

        perm = np.random.permutation(len(x))
        x_train = x[perm[:-10000]]
        t_train = t[perm[:-10000]]
        x_test = x[perm[-10000:]]
        t_test = t[perm[-10000:]]

        xp = x_train[t_train==1]
        X = Variable(np.array(xp, np.float32))
        g = model(X).data.T[0]
        xp_prob = 1/(1+np.exp(-g))
        xp_prob /= np.mean(xp_prob)
        xp_prob /= np.max(xp_prob)
        rand = np.random.uniform(size=len(xp))
        xp = xp[xp_prob > rand]
        perm = np.random.permutation(len(xp))
        xp = xp[perm[:pdata]]
        
        tp = np.ones(len(xp))
        tu = np.zeros(len(x_train))
        t_train = np.concatenate([tp, tu], axis=0)

        x_train = np.concatenate([xp, x_train], axis=0)

        model = normalNN(dim)
        optimizer = optimizers.Adam()
        optimizer.setup(model)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_stdr = NN_dr(dim).to(device)
        optimizer_stdr = optim.Adam(params=model_stdr.parameters(), lr = 1e-4)
        
        if gpu:
            gpu_device = 0
            cuda.get_device(gpu_device).use()
            model.to_gpu(gpu_device)
            xp = cuda.cupy
        else:
            xp = np

        t_test[t_test == -1] = 0
        model, optimizer, loss_list, acc1, acc2 = train_pu(x_train, t_train, x_test, t_test, pi, epoch, model, optimizer, batchsize, xp)
        model_dr, optimizer_dr, loss_list2, acc3 = train_dr(x_train, t_train, x_test, t_test, pi, epoch, model_stdr, optimizer_stdr, device, sym=True, tanh=True, batchsize=batchsize)        

        loss_pu[i] = loss_list
        est_error_pu[i] = acc1
        est_error_pubp[i] = acc2
        est_error_dr[i] = acc3        

        print(acc1[-1])
        print(acc2[-1])
        print(acc3[-1])

        seed += 1

        np.savetxt('loss_pu_mnist_%d.csv'%seed, loss_pu, delimiter=',')
        np.savetxt('est_error_pu_mnist_%d.csv'%seed, est_error_pu, delimiter=',')
        np.savetxt('est_error_pubp_mnist_%d.csv'%seed, est_error_pubp, delimiter=',')
        np.savetxt('est_error_dr_mnist_%d.csv'%seed, est_error_dr, delimiter=',')


    loss_pu_mean = np.mean(loss_pu, axis=0)
    est_error_pu_mean = np.mean(est_error_pu, axis=0)
    est_error_pubp_mean = np.mean(est_error_pubp, axis=0)
    est_error_pu_std = np.std(est_error_pu, axis=0)
    est_error_pubp_std = np.std(est_error_pubp, axis=0)
    return loss_pu_mean, est_error_pu_mean, est_error_pubp_mean, est_error_pu_std, est_error_pubp_std 

def main():
    loss_pu_mean, est_error_pu_mean, est_error_pubp_mean, est_error_pu_std, est_error_pubp_std = experiment()
    print(loss_pu_mean)
    print(est_error_pu_mean)
    print(est_error_pubp_mean)

if __name__ == "__main__":
    main()
