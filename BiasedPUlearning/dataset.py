import numpy as np

import chainer
from sklearn.decomposition import PCA
from sklearn.datasets import load_svmlight_file

def make_data(pi, num=1, dim=100, pdata=400, udata=3000, test_data=2000, datatype='mnist', doPCA=True, seed=10):
    if datatype == 'mnist':
        return make_mnist(pi, num, dim, pdata, udata, test_data, seed)
    elif datatype == 'adult':
        return make_adult(pi, dim, pdata, udata, test_data, doPCA, seed)
    elif datatype == 'ups':
        return make_adult(pi, dim, pdata, udata, test_data, doPCA, seed)
    elif datatype == 'mushrooms':
        return make_mushrooms(pi, dim, pdata, udata, test_data, doPCA, seed)

def make_mushrooms(pi, dim=100, pdata=400, udata=3000, test_data=2000, doPCA=True, seed=10):
    np.random.seed(seed)
    xs, ts = load_svmlight_file("dataset/mushrooms.txt")
    xs = xs.toarray()
    perm = np.random.permutation(len(xs))
    xs, xs_test = xs[perm[:6000]], xs[perm[6000:]]
    ts, ts_test = ts[perm[:6000]], ts[perm[6000:]]
    
    train_positive = np.int(np.round(udata*pi))
    train_negative = np.int(np.round(udata*(1-pi)))
    test_positive = np.int(np.round(test_data*pi))
    test_negative = np.int(np.round(test_data*(1-pi)))
    
    if doPCA is True:

        X = np.append(xs, xs_test, axis=0)
        pca = PCA(n_components=dim)
        pca.fit(X.T)
        X = pca.components_.T

        xs, xs_test = X[:len(xs)], X[len(xs):]

    xp_or = xs[ts == 1]
    xn_or = xs[ts == 2]
    xp_test = xs_test[ts_test==1]
    xn_test = xs_test[ts_test==2]

    n_p = len(xp_or)
    n_n = len(xn_or)

    perm_p = np.random.permutation(n_p)
    perm_n = np.random.permutation(n_n)

    xup = xp_or[perm_p[:train_positive]]
    xun = xn_or[perm_n[:train_negative]]
    xu = np.append(xup, xun, axis=0)
    
    perm_p = np.random.permutation(n_p)
    xp = xp_or[perm_p[:pdata]]
    tp = np.ones(pdata)
    tu = np.zeros(udata)
    x_train = np.append(xp, xu, axis=0)

    print(xp.shape)
    print(xu.shape)

    t_train = np.append(tp, tu)

    xpte = xp_test[:test_positive]
    xnte = xn_test[:test_negative]
    tp = np.ones(len(xpte))
    tn = np.zeros(len(xnte))

    x_test = np.append(xpte, xnte, axis=0)
    t_test = np.append(tp, tn)

    print(xpte.shape)
    print(xnte.shape)

    return x_train, t_train, x_test, t_test

def make_ups(pi, dim=100, pdata=400, udata=3000, test_data=2000, doPCA=True, seed=10):
    np.random.seed(seed)
    x_train, t_train = load_svmlight_file('dataset/usps')
    x_train = x_train.toarray()
    x_test, t_test = load_svmlight_file('dataset/usps.t')
    x_test = x_test.toarray()
    xs = np.concatenate([x_train, x_test])
    ts = np.concatenate([t_train, t_test])
    ts[ts%2==0] = 0
    ts[ts%2==1] = 1
    print(np.mean(t))
    
    perm = np.random.permutation(len(xs))
    xs, xs_test = xs[perm[:6000]], xs[perm[6000:]]
    ts, ts_test = ts[perm[:6000]], ts[perm[6000:]]
    
    train_positive = np.int(np.round(udata*pi))
    train_negative = np.int(np.round(udata*(1-pi)))
    test_positive = np.int(np.round(test_data*pi))
    test_negative = np.int(np.round(test_data*(1-pi)))
    
    if doPCA is True:

        X = np.append(xs, xs_test, axis=0)
        pca = PCA(n_components=dim)
        pca.fit(X.T)
        X = pca.components_.T

        xs, xs_test = X[:len(xs)], X[len(xs):]

    xp_or = xs[ts == 1]
    xn_or = xs[ts == 2]
    xp_test = xs_test[ts_test==1]
    xn_test = xs_test[ts_test==2]

    n_p = len(xp_or)
    n_n = len(xn_or)

    perm_p = np.random.permutation(n_p)
    perm_n = np.random.permutation(n_n)

    xup = xp_or[perm_p[:train_positive]]
    xun = xn_or[perm_n[:train_negative]]
    xu = np.append(xup, xun, axis=0)
    
    perm_p = np.random.permutation(n_p)
    xp = xp_or[perm_p[:pdata]]
    tp = np.ones(pdata)
    tu = np.zeros(udata)
    x_train = np.append(xp, xu, axis=0)

    print(xp.shape)
    print(xu.shape)

    t_train = np.append(tp, tu)

    xpte = xp_test[:test_positive]
    xnte = xn_test[:test_negative]
    tp = np.ones(len(xpte))
    tn = np.zeros(len(xnte))

    x_test = np.append(xpte, xnte, axis=0)
    t_test = np.append(tp, tn)

    print(xpte.shape)
    print(xnte.shape)

    return x_train, t_train, x_test, t_test

def make_adult(pi, dim=100, pdata=400, udata=3000, test_data=2000, doPCA=True, seed=10):
    np.random.seed(seed)
    xs, ts = load_svmlight_file("dataset/a9a.txt")
    xs = xs.toarray()
    perm = np.random.permutation(len(xs))
    xs, xs_test = xs[perm[:20000]], xs[perm[20000:]]
    ts, ts_test = ts[perm[:20000]], ts[perm[20000:]]
    
    train_positive = np.int(np.round(udata*pi))
    train_negative = np.int(np.round(udata*(1-pi)))
    test_positive = np.int(np.round(test_data*pi))
    test_negative = np.int(np.round(test_data*(1-pi)))
    
    if doPCA is True:

        X = np.append(xs, xs_test, axis=0)
        pca = PCA(n_components=dim)
        pca.fit(X.T)
        X = pca.components_.T

        xs, xs_test = X[:len(xs)], X[len(xs):]

    xp_or = xs[ts == 1]
    xn_or = xs[ts == -1]
    xp_test = xs_test[ts_test==1]
    xn_test = xs_test[ts_test==-1]

    n_p = len(xp_or)
    n_n = len(xn_or)

    perm_p = np.random.permutation(n_p)
    perm_n = np.random.permutation(n_n)

    xup = xp_or[perm_p[:train_positive]]
    xun = xn_or[perm_n[:train_negative]]
    xu = np.append(xup, xun, axis=0)
    
    perm_p = np.random.permutation(n_p)
    xp = xp_or[perm_p[:pdata]]
    tp = np.ones(pdata)
    tu = np.zeros(udata)
    x_train = np.append(xp, xu, axis=0)

    print(xp.shape)
    print(xu.shape)

    t_train = np.append(tp, tu)

    xpte = xp_test[:test_positive]
    xnte = xn_test[:test_negative]
    tp = np.ones(len(xpte))
    tn = np.zeros(len(xnte))

    x_test = np.append(xpte, xnte, axis=0)
    t_test = np.append(tp, tn)

    print(xpte.shape)
    print(xnte.shape)

    return x_train, t_train, x_test, t_test
    
    
def make_mnist(pi, num, dim=100, pdata=400, udata=3000, test_data=2000, seed=10):
    np.random.seed(seed)
    train, test = chainer.datasets.get_mnist()
    xs, ts = train._datasets
    xs_test, ts_test = test._datasets

    train_positive = np.int(np.round(udata*pi))
    train_negative = np.int(np.round(udata*(1-pi)))
    test_positive = np.int(np.round(test_data*pi))
    test_negative = np.int(np.round(test_data*(1-pi)))
    
    if doPCA is True:

        X = np.append(xs, xs_test, axis=0)
        pca = PCA(n_components=dim)
        pca.fit(X.T)
        X = pca.components_.T

        xs, xs_test = X[:len(xs)], X[len(xs):]

    xp_or = xs[ts == num]
    xn_or = xs[ts == 0]
    xp_test = xs_test[ts_test==num]
    xn_test = xs_test[ts_test==0]

    n_p = len(xp_or)
    n_n = len(xn_or)

    perm_p = np.random.permutation(n_p)
    perm_n = np.random.permutation(n_n)

    xup = xp_or[perm_p[:train_positive]]
    xun = xn_or[perm_n[:train_negative]]
    xu = np.append(xup, xun, axis=0)
    
    perm_p = np.random.permutation(n_p)
    xp = xp_or[perm_p[:pdata]]
    tp = np.ones(pdata)
    tu = np.zeros(udata)
    x_train = np.append(xp, xu, axis=0)

    print(xp.shape)
    print(xu.shape)

    t_train = np.append(tp, tu)

    xpte = xp_test[:test_positive]
    xnte = xn_test[:test_negative]
    tp = np.ones(len(xpte))
    tn = np.zeros(len(xnte))

    x_test = np.append(xpte, xnte, axis=0)
    t_test = np.append(tp, tn)

    print(xpte.shape)
    print(xnte.shape)

    return x_train, t_train, x_test, t_test