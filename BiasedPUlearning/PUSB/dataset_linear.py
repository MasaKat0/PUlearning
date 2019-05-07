import numpy as np

import chainer
from sklearn.decomposition import PCA
from sklearn.datasets import load_svmlight_file

def make_data(datatype='mnist', seed=2018, pca_dim=100):
    print("data_name", datatype)
    x, t, doPCA = get_data(datatype)
    print("x_shape", x.shape)
    print("t_shape", t.shape)
    #if doPCA is True:
        #pca = PCA(n_components=pca_dim)
        #pca.fit(x.T)
        #x = pca.components_.T
    return x, t

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
        
    elif datatype == "ijcnn1":
        x, t = load_svmlight_file('./dataset/ijcnn1')
        x = x.toarray()
        t[ ~(t == 1)] = 0
        print(np.mean(t))
        doPCA = False

    elif datatype == "w1a":
        x_train, t_train = load_svmlight_file('./dataset/w1a')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('./dataset/w1a.t')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[ ~(t == 1)] = 0
        print(np.mean(t))
        doPCA = False
              
    else:
        raise ValueError

    return x, t, doPCA
