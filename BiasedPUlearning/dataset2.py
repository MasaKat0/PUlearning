import numpy as np

import chainer
from sklearn.decomposition import PCA
from sklearn.datasets import load_svmlight_file

def make_data2(pi, udata=3000, datatype='mnist', flip=False, seed=2018, pca_dim=100):
    print("data_name", datatype)
    x, t, doPCA = get_data(datatype)
    print("x_shape", x.shape)
    print("t_shape", t.shape)

    if flip:
        ts = np.copy(t)
        t[ts == 1] = 0
        t[ts == 0] = 1

    return make_pu(x, t, pi, udata, doPCA, pca_dim, seed)

def get_data(datatype):
    doPCA = False
    if datatype == "mushroom":
        x, t = load_svmlight_file("dataset/mushrooms.txt")
        x = x.toarray()
        t[t == 1] = 0
        t[t == 2] = 1
        doPCA = True

    elif datatype == "waveform":
        data = np.loadtxt('dataset/waveform.txt', delimiter=',')
        x, t = data[:, :-1], data[:, -1]
        t[t == 2] = 0

    elif datatype == "shuttle":
        x_train, t_train = load_svmlight_file('dataset/shuttle.scale.txt')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('dataset/shuttle.scale.t.txt')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[ ~(t == 1)] = 0
        
    elif datatype == "pageblocks":
        data = np.loadtxt('dataset/page-blocks.txt')
        x, t = data[:, :-1], data[:, -1]
        t[~(t == 1)] = 0

    elif datatype == "digits":
        train, test = chainer.datasets.get_mnist()
        x_train, t_train = train._datasets
        x_test, t_test = test._datasets
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[t%2==0] = 0
        t[t%2==1] = 1
        doPCA = True

    elif datatype == "spambase":
        data = np.loadtxt('dataset/spambase.data.txt', delimiter=',')
        x, t = data[:, :-1], data[:, -1]
        
    elif datatype == "usps":
        x_train, t_train = load_svmlight_file('dataset/usps')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('dataset/usps.t')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[t%2==0] = 0
        t[t%2==1] = 1
        print(np.mean(t))
        doPCA = True
        
    elif datatype == "connect-4":
        x, t = load_svmlight_file('dataset/connect-4.txt')
        x = x.toarray()
        t[t == -1] = 0
        print(np.mean(t))
        doPCA = True
        
    elif datatype == "protein":
        x_train, t_train = load_svmlight_file('dataset/protein.txt')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('dataset/protein.t.txt')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[ ~(t == 1)] = 0
        print(np.mean(t))
        doPCA = True
              
    else:
        raise ValueError

    return x, t, doPCA

def make_pu(x, t, pi, udata, doPCA, pca_dim, seed):
    np.random.seed(seed)
    N = len(x)

    train_positive = np.int(np.round(udata*pi))
    train_negative = np.int(np.round(udata*(1-pi)))
    
    if doPCA is True:

        pca = PCA(n_components=pca_dim)
        pca.fit(x.T)
        x = pca.components_.T

    xp_or = x[t == 1]
    xn_or = x[t == 0]

    n_p = len(xp_or)
    n_n = len(xn_or)

    perm_p = np.random.permutation(n_p)
    if len(perm_p) < train_positive:
        perm_p = np.append(perm_p, np.random.choice(perm_p,train_positive-len(perm_p))) 
    xp = xp_or[perm_p[:]]

    perm_p = np.random.permutation(n_p)
    if len(perm_p) < train_positive:
        perm_p = np.append(perm_p, np.random.choice(perm_p,train_positive-len(perm_p))) 
    perm_n = np.random.permutation(n_n)
    if len(perm_n) < train_negative:
        perm_n = np.append(perm_n, np.random.choice(perm_n,train_negative-len(perm_n)))
                           
    xup = xp_or[perm_p[:train_positive]]
    xun = xn_or[perm_n[:train_negative]]
    xu = np.append(xup, xun, axis=0)
    
    tp = np.ones(pdata)
    tu = np.zeros(udata)

    x = np.append(xp, xu, axis=0)
    t = np.append(tp, tu)

    print("xp_shape", xp.shape)
    print("xu_shape", xu.shape)
    print("t_shape", t.shape)

    return x, t