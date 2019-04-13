import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from dataset2 import make_data2
from dataset3 import make_data3
from pubp import PU
from densratio import densratio

def experiment(datatype, udata):
    priors = [0.2, 0.4, 0.6, 0.8]
    ite = 100
    pca_dim = 100
    pdata = 400
    num_basis = 300
    lda_chosen = 0.01

    seed = 2018

    est_error_pu = np.zeros((len(udata), len(priors), ite))
    est_error_pubp = np.zeros((len(udata), len(priors), ite))
    est_error_dr = np.zeros((len(udata), len(priors), ite))

    for i in range(len(udata)):
        u = udata[i]
        for j in range(len(priors)):
            pi = priors[j]
            for k in range(ite):
                np.random.seed(seed)
                #PN classification
                x, t = make_data3(datatype=datatype)
                x = x/np.max(x, axis=0)
                one = np.ones((len(x),1))
                x_pn = np.concatenate([x, one], axis=1)
                classifier = LogisticRegression(C=0.01, penalty='l2')
                classifier.fit(x_pn, t) 
                                
                perm = np.random.permutation(len(x))
                x_train = x[perm[:-3000]]
                t_train = t[perm[:-3000]]
                x_test = x[perm[-3000:]]
                t_test = t[perm[-3000:]]
                
                xp = x_train[t_train==1]
                one = np.ones((len(xp),1))
                xp_temp = np.concatenate([xp, one], axis=1)
                xp_prob = classifier.predict_proba(xp_temp)[:,1]
                xp_prob /= np.mean(xp_prob)
                xp_prob = xp_prob**20
                xp_prob /= np.max(xp_prob)
                rand = np.random.uniform(size=len(xp))
                xp = xp[xp_prob > rand]
                perm = np.random.permutation(len(xp))
                xp = xp[perm[:pdata]]
                
                updata = np.int(u*pi)
                undata = u - updata
                
                xp_temp = x_train[t_train==1]
                xn_temp = x_train[t_train==0]
                perm = np.random.permutation(len(xp_temp))
                xp_temp = xp_temp[perm[:updata]]
                
                perm = np.random.permutation(len(xn_temp))
                xn_temp = xn_temp[perm[:undata]]
                
                xu = np.concatenate([xp_temp, xn_temp], axis=0)
                
                x = np.concatenate([xp, xu], axis=0)
                
                tp = np.ones(len(xp))
                tu = np.zeros(len(xu))
                t = np.concatenate([tp, tu], axis=0)
                
                updata = np.int(1000*pi)
                undata = 1000 - updata
                
                xp_test = x_test[t_test == 1]
                perm = np.random.permutation(len(xp_test))
                xp_test = xp_test[perm[:updata]]
                xn_test = x_test[t_test == 0]
                perm = np.random.permutation(len(xn_test))
                xn_test = xn_test[perm[:undata]]
                
                x_test = np.concatenate([xp_test, xn_test], axis=0)
                tp = np.ones(len(xp_test))
                tu = np.zeros(len(xn_test))
                t_test = np.concatenate([tp, tu], axis=0)
                
                pu = PU(pi=pi)
                one = np.ones((len(x),1))
                x_train = np.concatenate([x, one], axis=1)
                one = np.ones((len(x_test),1))
                x_test = np.concatenate([x_test, one], axis=1)
                
                res = pu.minimize(x_train, t, lda_chosen)
                acc1 = pu.test(x_test, res, t_test, quant=False)
                acc2 = pu.test(x_test, res, t_test, quant=True, pi=pi)
                
                result = densratio(x_train[t==1], x_train[t==0])
                r = result.compute_density_ratio(x_test)
                temp = np.copy(r)
                temp = np.sort(temp)
                theta = temp[np.int(np.floor(len(x_test)*(1-pi)))]
                pred = np.zeros(len(x_test))
                pred[r > theta] = 1
                acc3 = np.mean(pred == t_test)
                
                est_error_pu[i, j, k] = acc1
                est_error_pubp[i, j, k] = acc2
                est_error_dr[i, j, k] = acc3

                seed += 1
                
                print(acc1)
                print(acc2)
                print(acc3)
    
    est_error_pu_mean = np.mean(est_error_pu, axis=2)
    est_error_pubp_mean = np.mean(est_error_pubp, axis=2)
    est_error_dr_mean = np.mean(est_error_dr, axis=2)
    est_error_pu_std = np.std(est_error_pu, axis=2)
    est_error_pubp_std = np.std(est_error_pubp, axis=2)
    est_error_dr_std = np.std(est_error_dr, axis=2)
    return est_error_pu_mean, est_error_pubp_mean, est_error_pu_std, est_error_pubp_std, est_error_dr_std, est_error_dr_std

def main():
    datasets = ["usps", "connect-4", "spambase"]
    udata = [800, 1600, 3200]
    priors = [0.2, 0.4, 0.6, 0.8]

    for d in datasets:
        est_error_pu_mean, est_error_pubp_mean, est_error_pu_std, est_error_pubp_std, est_error_dr_mean, est_error_dr_std = experiment(d, udata)
        est_error_pu_mean = 1 - est_error_pu_mean
        est_error_pubp_mean = 1 - est_error_pubp_mean
        est_error_dr_mean = 1 - est_error_dr_mean
        est_error_pu = np.concatenate([np.array([est_error_pu_mean[0,:]]).T, np.array([est_error_pu_std[0,:]]).T, \
            np.array([est_error_pu_mean[1,: ]]).T, np.array([est_error_pu_std[1,:]]).T, \
            np.array([est_error_pu_mean[2,:]]).T, np.array([est_error_pu_std[2,:]]).T], axis=1)
        est_error_pubp = np.concatenate([np.array([est_error_pubp_mean[0,:]]).T, np.array([est_error_pubp_std[0,:]]).T, \
                np.array([est_error_pubp_mean[1,:]]).T, np.array([est_error_pubp_std[1,:]]).T, \
                np.array([est_error_pubp_mean[2,:]]).T, np.array([est_error_pubp_std[2,:]]).T], axis=1)
        est_error_dr = np.concatenate([np.array([est_error_dr_mean[0,:]]).T, np.array([est_error_dr_std[0,:]]).T, \
                np.array([est_error_dr_mean[1,:]]).T, np.array([est_error_dr_std[1,:]]).T, \
                np.array([est_error_dr_mean[2,:]]).T, np.array([est_error_dr_std[2,:]]).T], axis=1)
        est_error_pu = pd.DataFrame(est_error_pu)
        est_error_pu["priors"] = priors
        est_error_pu = est_error_pu.set_index("priors")
        est_error_pu.columns = ["mean", "std", "mean", "std", "mean", "std"]
        est_error_pu = est_error_pu.T
        est_error_pu["statistics"] = list(est_error_pu.index)
        est_error_pu["num of udata"] = [ 800, 800, 1600, 1600, 3200, 3200]
        est_error_pu = est_error_pu.set_index(["num of udata", "statistics"])
        est_error_pu = est_error_pu.T

        est_error_pubp = pd.DataFrame(est_error_pubp)
        est_error_pubp["priors"] = priors
        est_error_pubp = est_error_pubp.set_index("priors")
        est_error_pubp.columns = ["mean", "std", "mean", "std", "mean", "std"]
        est_error_pubp = est_error_pubp.T
        est_error_pubp["statistics"] = list(est_error_pubp.index)
        est_error_pubp["num of udata"] = [800, 800, 1600, 1600, 3200, 3200]
        est_error_pubp = est_error_pubp.set_index(["num of udata", "statistics"])
        est_error_pubp = est_error_pubp.T
        est_error_dr = pd.DataFrame(est_error_dr)
        est_error_dr["priors"] = priors
        est_error_dr = est_error_dr.set_index("priors")
        est_error_dr.columns = ["mean", "std", "mean", "std", "mean", "std"]
        est_error_dr = est_error_dr.T
        est_error_dr["statistics"] = list(est_error_dr.index)
        est_error_dr["num of udata"] = [800, 800, 1600, 1600, 3200, 3200]
        est_error_dr = est_error_dr.set_index(["num of udata", "statistics"])
        est_error_dr = est_error_dr.T

        est_error_pu.to_csv("%s_pu.csv"%d)
        est_error_pubp.to_csv("%s_pubp.csv"%d)
        est_error_dr.to_csv("%s_dr.csv"%d)
        print(est_error_pu)
        print(est_error_pubp)
        print(est_error_dr)

if __name__ == '__main__':
    main()
