from __future__ import division
import os, re, math
from collections import Counter
import numpy as np
import pandas as pd
#from sklearn.decomposition import PCA
import networkx as nx
import scipy.stats as stats
import matplotlib.pyplot as plt


#df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
#df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
#df_delta = pt.likelihood_matrix(df, 'Tenaillon_et_al').get_likelihood_matrix()
#X = pt.hellinger_transform(df_delta)
#pca = PCA()
#df_out = pca.fit_transform(X)




def get_ba_cov_matrix(n_genes, cov, cov2 = None, prop=False, m=2, get_node_edge_sum=False, rho=None, rho2=None):
    '''Based off of Gershgorin circle theorem, we can expect
    that the code will eventually produce mostly matrices
    that aren't positive definite as the covariance value
    increases and/or more edges added to incidence matrix'''
    while True:
        if rho == None:
            ntwk = nx.barabasi_albert_graph(n_genes, m)
            ntwk_np = nx.to_numpy_matrix(ntwk)
        else:
            ntwk_np, rho_estimate = get_correlated_rndm_ntwrk(n_genes, m=m, rho=rho, count_threshold = 10000)

        if cov2 == None:
            C = ntwk_np * cov
        else:
            C = ntwk_np * cov
            C2 = ntwk_np * cov2
            np.fill_diagonal(C2, 1)

        np.fill_diagonal(C, 1)

        if prop == True and cov2 == None:
            diag_C = np.tril(C, k =-1)
            i,j = np.nonzero(diag_C)
            ix = np.random.choice(len(i), int(np.floor((1-prop) * len(i))), replace=False)
            C[np.concatenate((i[ix],j[ix]), axis=None), np.concatenate((j[ix],i[ix]), axis=None)] = -1*cov

        if cov2 == None:
            if np.all(np.linalg.eigvals(C) > 0) == True:
                if rho == None:
                    return C
                else:
                    return C, rho_estimate

        else:
            if (np.all(np.linalg.eigvals(C) > 0) == True) and (np.all(np.linalg.eigvals(C2) > 0) == True):
                return C, C2

def get_pois_sample(lambda_, u):
    x = 0
    p = math.exp(-lambda_)
    s = p
    #u = np.random.uniform(low=0.0, high=1.0)
    while u > s:
         x = x + 1
         p  = p * lambda_ / x
         s = s + p
    return x


def get_count_pop(lambdas, C):
    mult_norm = np.random.multivariate_normal(np.asarray([0]* len(lambdas)), C)#, tol=1e-6)
    mult_norm_cdf = stats.norm.cdf(mult_norm)
    counts = [ get_pois_sample(lambdas[i], mult_norm_cdf[i]) for i in range(len(lambdas))  ]

    return np.asarray(counts)



def get_sim_matrix(G, N, cov):
    #df_out=open(out_name, 'w')
    #df_out.write('\t'.join(['N', 'G', 'Cov', 'Iteration', 'dist_percent', 'z_score']) + '\n')
    #for i in range(iter1):
    C = get_ba_cov_matrix(G, cov)
    while True:
        lambda_genes = np.random.gamma(shape=1, scale=1, size=G)
        test_cov = np.stack( [get_count_pop(lambda_genes, C= C) for x in range(N)] , axis=0 )
        if (np.any(test_cov.sum(axis=1) == 0 )) == False:
            break

    return test_cov
    # check and remove empty columns
    #test_cov = test_cov[:, ~np.all(test_cov == 0, axis=0)]
    #euc_percent, z_score = pt.matrix_vs_null_one_treat(test_cov, iter2)



def get_ev_spacings(array):
    spacings = []
    for i in range(len(array) -1):
        s_i = array[i+1] - array[i]
        spacings.append(s_i)
    spacings_np = np.asarray(spacings)
    spacings_np = spacings_np / np.mean(spacings_np)
    return spacings_np


def get_e_values(N, G, cov, iter):
    spacings_all = []
    max_ev = []
    for i in range(iter):
        if i % 100 == 0:
            print("Iteration " + str(i))
        X = get_sim_matrix(100, N, 0.2)
        X_centered = X - np.mean(X, axis=0)
        # N-1 because we calculated the mean
        cov = np.dot(X_centered.T, X_centered) / (N-1)
        ev , eig = np.linalg.eig(cov)
        ev.sort()
        ev = ev[1:]
        spacings = get_ev_spacings(ev)
        spacings_all.extend(spacings)
        max_ev.append(ev[-1])

    max_ev = np.asarray(max_ev)
    mean = (math.sqrt(N-1) + math.sqrt(G)) ** 2
    std_dev = (math.sqrt(N-1) + math.sqrt(G)) * (((1/math.sqrt(N-1)) +  (1/math.sqrt(G))) ** (1/3) )
    scaled_max_ev = (max_ev - mean) / std_dev

    return scaled_max_ev


N = 100
G = 200
iter=5000
pos_cov_space = get_e_values(N, G, 0.2, iter)
neg_cov_space = get_e_values(N, G, -0.2, iter)
indep_space = get_e_values(N, G, 0, iter)

s_poisson = np.linspace(min(indep_space), 10, num=1000)
P_s_poisson = np.exp(-1* s_poisson)

fig = plt.figure()
#plt.plot(s_poisson, P_s_poisson)
plt.hist(indep_space, density=True, bins=100, alpha = 0.5, color='#FFA500', label="Cov=0")
plt.hist(neg_cov_space, density=True, bins=100, alpha = 0.5, color='#FF6347', label="Cov=-0.2")
plt.hist(pos_cov_space, density=True, bins=100, alpha = 0.5, color='#87CEEB', label="Cov=0.2")

#plt.xlim(0, 6)
plt.legend(loc='upper right', fontsize=14)
#plt.xlabel("Eigenvalue spacing", fontsize = 18)
plt.xlabel("Maximum Eigenvalue", fontsize = 18)
plt.ylabel("Frequency", fontsize = 18)
fig.tight_layout()
plot_out = os.path.expanduser("~/GitHub/NetworkEvolution") + '/figs/test_eigen_shape.png'
fig.savefig(plot_out, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
