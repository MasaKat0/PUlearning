import numpy as np
import six
from scipy import optimize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def train_dr(x_train, t_train, x_test, t_test, pi, epoch, model, optimizer, device, sym=True, tanh=True, batchsize=5000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = len(x_train)
    loss_list = []
    acc_list = []

    for ep in range(1, epoch+1):
        
        perm = np.random.permutation(N)

        loss_step = 0
        count = 0
        f = np.array([])
        for i in six.moves.range(0, N, batchsize):
            model.train()
            optimizer.zero_grad()
            x = x_train[perm[i:i + batchsize]]
            x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)
            ts = np.array([t_train[perm[i:i + batchsize]]]).T
            t0 = torch.tensor(ts, dtype=torch.float32)
            t0 = t0.to(device)
            t1 = torch.tensor(1-ts, dtype=torch.float32)
            t1 = t1.to(device)
            
            tanh = False            
            if tanh:
                densratio = torch.tanh(model(x))*4
            else:
                densratio = model(x)
            densratio1 = torch.exp(densratio)
            loss1 = loss_func(densratio1, t0, t1, flip=False)
            if sym:
                densratio2 = torch.exp(-densratio)
                loss2 = loss_func(densratio2, t0, t1, flip=True)
                l1 = np.float64(loss1.data)
                l2 = np.float64(loss2.data)
                if l1 > l2:
                    loss = loss1
                else:
                    loss = loss2
                #loss = loss1 + loss2
            else:
                loss = loss1
            #print("running loss 1", np.float64(loss.data))
            if (np.sum(ts) < len(ts)) and (np.sum(ts) > 0):
                loss.backward()
                optimizer.step()
                count += 1
                loss_step += np.float64(loss.data)
                #print("running loss 2", np.float64(loss.data))
                #loss_step += test_pu(x_train, t_train, model, pi, device, batchsize=batchsize)
            densratio_value = np.float64(densratio1.data).T[0]
            f = np.append(f, densratio_value[ts.T[0]==0], axis=0)
            #print(densratio_value)
        loss_step /= count
        model.eval()
        #loss_step = test_dr(x_train, t_train, model, pi, device, batchsize=batchsize)
        loss_list.append(loss_step)
        print("running loss", loss_step)       
        #pi = 1
        acc = test_dr(x_test, t_test, model, pi, device, batchsize=batchsize)
        acc_list.append(acc)
        print("running test", acc)

    loss_list = np.array(loss_list)
    acc_list = np.array(acc_list)
    return model, optimizer, loss_list, acc_list

def test_dr(xt, tt, model, pi, device, batchsize=100):
    f = np.array([])
    for i in six.moves.range(0, len(xt), batchsize):
        xs = xt[i:i + batchsize]
        x = torch.tensor(xs, dtype=torch.float32)
        x = x.to(device)

        #densratio = torch.exp(torch.tanh(model(x))*3)
        densratio = torch.exp(model(x))
        densratio_value = np.float64(densratio.data).T[0]
        f = np.append(f, densratio_value, axis=0)
    
    theta = 0
    temp = np.copy(f)
    temp = np.sort(temp)
    theta = temp[np.int(np.floor(len(x)*(1-pi)))]
    pred = np.zeros(len(xt))
    pred[f > theta] = 1
    acc = np.mean(pred == tt) 
    return acc

def loss_func(densratio, t0, t1, flip=True):
    if flip:
        g1 = densratio*t0
        g1 = (g1**2).sum()/(2*t0.sum())
       
        g10 = densratio*t1
        g10 = g10.sum()/(t1.sum())

    else:
        g1 = densratio*t1
        g1 = (g1**2).sum()/(2*t1.sum())

        g10 = densratio*t0
        g10 = g10.sum()/(t0.sum())
    
    return g1 - g10
