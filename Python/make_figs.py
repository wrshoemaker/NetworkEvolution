from __future__ import division
import math, os, re
import numpy as np
import pandas as pd
import network_tools as nt
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table
from collections import Counter
#import scipy.stats
from scipy.stats import binom





def fig4():
    df_path = nt.get_path() + '/data/Tenaillon_et_al/network.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)

    #df_C_path = nt.get_path() + '/data/Tenaillon_et_al/network_CCs.txt'
    df_C_path = nt.get_path() + '/data/Tenaillon_et_al/network_CCs_no_kmax.txt'
    df_C = pd.read_csv(df_C_path, sep = '\t', header = 'infer', index_col = 0)
    kmax_df = max(df_C.k_i.values)
    mean_C_df = np.mean(df_C.loc[df_C['k_i'] >= 2].C_i.values)
    df_null_path = nt.get_path() + '/data/Tenaillon_et_al/permute_network.txt'
    df_null = pd.read_csv(df_null_path, sep = '\t', header = 'infer', index_col = 0)

    df_no_max = df.copy()
    df_no_max = df_no_max.drop('kpsD', axis=0)
    df_no_max = df_no_max.drop('kpsD', axis=1)
    #dist_df = nt.networkx_distance(df)
    dist_df = nt.networkx_distance(df_no_max)

    C_mean_null = df_null.C_mean_no1or2.tolist()
    C_mean_null = [x for x in C_mean_null if str(x) != 'nan']
    d_mean_null = df_null.d_mean.tolist()
    k_max_null = df_null.k_max.tolist()

    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    k_list = []
    for index, row in df.iterrows():
        k_row = sum(i != 0 for i in row.values) - 1
        if k_row > 0:
            k_list.append(k_row)

    count_k_list = Counter(k_list)
    count_k_list_sum = sum(count_k_list.values())
    count_k_list_x = list(count_k_list.keys())
    count_k_list_y = [(i / count_k_list_sum) for i in count_k_list.values()]

    k_list_no_max = []
    for index, row in df_no_max.iterrows():
        k_row = sum(i != 0 for i in row.values) - 1
        if k_row > 0:
            k_list_no_max.append(k_row)

    count_k_list_no_max = Counter(k_list_no_max)
    count_k_list_sum_no_max = sum(count_k_list_no_max.values())
    count_k_list_x_no_max = list(count_k_list_no_max.keys())
    count_k_list_y_no_max = [(i / count_k_list_sum_no_max) for i in count_k_list_no_max.values()]

    ax1.scatter(count_k_list_x, count_k_list_y, marker = "o", edgecolors='#244162', c = '#175ac6', alpha = 0.4, s = 60, zorder=4)
    # red colors
    # edge #C92525
    # c #FF4343
    #ax1.scatter(count_k_list_x_no_max, count_k_list_y_no_max, marker = "o", edgecolors='#C92525', c = '#FF4343', alpha = 0.4, s = 60, zorder=4)

    count_k_list_x.sort()
    #m = 0.56086623
    #pred_y = [ ((2 * m * (m+1)) / (j * (j+1) * (j+2) )) for j in count_k_list_x ]
    #ax1.plot(count_k_list_x, pred_y, c = 'k', lw = 2.5,
    #    ls = '--', zorder=2)
    p = sum(k_list) / (((df.shape[0]) * (df.shape[0]-1)) / 2)
    p_no_max = sum([i for i in k_list if i != 181])  /  (((df.shape[0]-1) * (df.shape[0]-2)) / 2)

    binom_x = np.arange(0, max(k_list))
    binom_y = binom.pmf(binom_x, df.shape[0] - 1, p)
    binom_y_noMax = binom.pmf(binom_x, df.shape[0] - 2, p_no_max)

    ax1.plot(binom_x, binom_y_noMax, c = 'k', lw = 2.5, ls = '--', zorder=2)
    ax1.set_xlim([0.5, 400])
    ax1.set_ylim([0.001, 1])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$k_{i}$', fontsize = 14)
    ax1.set_ylabel("Frequency", fontsize = 14)


    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    ax2.hist(k_max_null,bins=30, weights=np.zeros_like(k_max_null) + 1. / len(k_max_null), alpha=0.8, color = '#175ac6')
    #ax2.axvline(max(k_list_no_max), color = 'red', lw = 2, ls = ':')
    ax2.axvline(max(k_list_no_max), color = 'red', lw = 2, ls = '--')
    #ax2.axvline(kmax_df, color = 'red', lw = 2, ls = '--')
    #ax2.set_xscale('log')
    ax2.set_xlabel(r'$k_{max}$', fontsize = 14)
    ax2.set_ylabel("Frequency", fontsize = 14)

    k_max_null.append(kmax_df)
    relative_position_k_max = sorted(k_max_null).index(kmax_df) / (len(k_max_null) -1)
    if relative_position_k_max > 0.5:
        p_score_k_max = 1 - relative_position_k_max
    else:
        p_score_k_max = relative_position_k_max
    print('kmax p-score = ' + str(round(p_score_k_max, 3)))
    #ax2.text(0.366, 0.088, r'$p < 0.05$', fontsize = 10)

    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    #print(C_mean_null)
    ax3.hist(C_mean_null,bins=30, weights=np.zeros_like(C_mean_null) + 1. / len(C_mean_null), alpha=0.8, color = '#175ac6')
    ax3.axvline(mean_C_df, color = 'red', lw = 2, ls = '--')
    #ax3.set_xlabel("Mean clustering coefficient", fontsize = 14)
    ax3.set_xlabel('Mean clustering coefficient, ' + r'$\left \langle C \right \rangle$', fontsize = 14)
    ax3.set_ylabel("Frequency", fontsize = 14)

    C_mean_null.append(mean_C_df)
    relative_position_mean_C = sorted(C_mean_null).index(mean_C_df) / (len(C_mean_null) -1)
    if relative_position_mean_C > 0.5:
        p_score_mean_C = 1 - relative_position_mean_C
    else:
        p_score_mean_C = relative_position_mean_C
    print('mean C p-score = ' + str(round(p_score_mean_C, 3)))
    #ax3.text(0.078, 0.115, r'$p < 0.05$', fontsize = 10)


    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax4.hist(d_mean_null,bins=30, weights=np.zeros_like(d_mean_null) + 1. / len(d_mean_null), alpha=0.8, color = '#175ac6')
    ax4.axvline(dist_df, color = 'red', lw = 2, ls = '--')
    #ax4.set_xlabel("Mean distance", fontsize = 14)
    ax4.set_xlabel('Mean distance, ' + r'$\left \langle d \right \rangle$', fontsize = 14)
    ax4.set_ylabel("Frequency", fontsize = 14)

    d_mean_null.append(dist_df)
    relative_position_d_mean = sorted(d_mean_null).index(dist_df) / (len(d_mean_null) -1)
    if relative_position_d_mean > 0.5:
        p_score_d_mean = 1 - relative_position_d_mean
    else:
        p_score_d_mean = relative_position_d_mean
    print('mean pairwise distance p-score = ' + str(round(p_score_d_mean, 3)))
    #ax4.text(89.1, 0.09, r'$p \nless  0.05$', fontsize = 10)

    plt.tight_layout()
    fig_name = nt.get_path() + '/figs/fig4.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



# def fig5():
# effect of sample size on featyres




def fig6():
    network_dir = nt.get_path() + '/data/Good_et_al/networks_naive/'
    #network_dir = nt.get_path() + '/data/Good_et_al/networks_BIC/'
    time_nodes = []
    time_kmax = []
    for filename in os.listdir(network_dir):
        if filename == '.DS_Store':
            continue
        df = pd.read_csv(network_dir + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        time_nodes.append((int(time), df.shape[0]))
        time_kmax.append((int(time), max(df.astype(bool).sum(axis=0).values)))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    time_nodes_sorted = sorted(time_nodes, key=lambda tup: tup[0])
    x_nodes = [i[0] for i in time_nodes_sorted]
    y_nodes = [i[1] for i in time_nodes_sorted]
    ax1.scatter(x_nodes, y_nodes, marker = "o", edgecolors='#244162', \
        c = '#175ac6', s = 80, zorder=3, alpha = 0.6)
    ax1.set_xlabel("Time (generations)", fontsize = 14)
    ax1.set_ylabel('Network size, ' + r'$N$', fontsize = 14)
    ax1.set_ylim(5, 500)
    #ax1.set_yscale('log')


    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    time_kmax_sorted = sorted(time_kmax, key=lambda tup: tup[0])
    x_kmax = [i[0] for i in time_kmax_sorted]
    y_kmax = [i[1] for i in time_kmax_sorted]
    x_kmax = np.log10(x_kmax)
    y_kmax = np.log10(y_kmax)

    '''The below regression code is from the GitHub repository
    ScalingMicroBiodiversity and is licensed under a
    GNU General Public License v3.0.

    https://github.com/klocey/ScalingMicroBiodiversity
    '''
    df_regression = pd.DataFrame({'t': list(x_kmax)})
    df_regression['kmax'] = list(y_kmax)
    f = smf.ols('kmax ~ t', df_regression).fit()

    R2 = f.rsquared
    pval = f.pvalues
    intercept = f.params[0]
    slope = f.params[1]
    X = np.linspace(min(x_kmax), max(x_kmax), 1000)
    Y = f.predict(exog=dict(t=X))
    print(min(x_kmax), max(y_kmax))

    st, data, ss2 = summary_table(f, alpha=0.05)
    fittedvalues = data[:,2]
    pred_mean_se = data[:,3]
    pred_mean_ci_low, pred_mean_ci_upp = data[:,4:6].T
    pred_ci_low, pred_ci_upp = data[:,6:8].T

    slope_to_gamme = (1/slope) + 1

    ax2.scatter([10**i for i in x_kmax], [10**i for i in y_kmax], c='#175ac6', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.6, zorder=1)#, edgecolors='none')
    ax2.fill_between([10**i for i in x_kmax], [10**i for i in pred_ci_low], [10**i for i in pred_ci_upp], color='#175ac6', lw=0.5, alpha=0.2, zorder=2)
    ax2.text(250, 100, r'$k_{max}$'+ ' = ' + str(round(10**intercept,2)) + '*' + r'$t^ \frac{1}{\,' + str(round(slope_to_gamme,2)) + '- 1}$', fontsize=9, color='k', alpha=0.9)
    ax2.text(250, 60,  r'$r^2$' + ' = ' +str("%.2f" % R2), fontsize=9, color='0.2')
    ax2.plot([10**i for i in X.tolist()], [10**i for i in Y.tolist()], '--', c='k', lw=2, alpha=0.8, color='k', label='Power-law', zorder=2)
    ax2.set_xlabel("Time (generations)", fontsize = 14)
    ax2.set_ylabel(r'$k_{max}$', fontsize = 14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')


    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    df_net_feats_path = nt.get_path() + '/data/Good_et_al/network_naive_features.txt'
    #df_net_feats_path = nt.get_path() + '/data/Good_et_al/network_naive_features.txt'
    df_net_feats = pd.read_csv(df_net_feats_path, sep = '\t', header = 'infer')
    x_C = df_net_feats.N.values
    y_C = df_net_feats.C_mean.values

    ax3.scatter(x_C, y_C, c='#175ac6', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.6, zorder=1)
    x_C_range = list(range(10, max(x_C)))
    barabasi_albert_range_C = [  ((np.log(i) ** 2) / i) for i in x_C_range ]
    random_range_c = [ (1/i) for i in x_C_range ]

    x_C_sort = list(set(x_C.tolist()))
    x_C_sort.sort()
    model = nt.clusterBarabasiAlbert(x_C, y_C)
    b0_start = [0.01, 0.1, 1, 10]
    z_start = [-2,-0.5]
    results = []
    for b0 in b0_start:
        for z in z_start:
            start_params = [b0, z]
            result = model.fit(start_params = start_params)
            results.append(result)
    AICs = [result.aic for result in results]
    best = results[AICs.index(min(AICs))]
    best_CI_FIC = nt.CI_FIC(best)
    best_CI = best.conf_int()
    best_params = best.params

    barabasi_albert_range_C_ll = nt.cluster_BA(np.sort(x_C), best_params[0])

    ax3.plot(np.sort(x_C), barabasi_albert_range_C_ll, c = 'k', lw = 2.5,
        ls = '--', zorder=2)
    #plt.plot(x_C_range, random_range_c, c = 'r', lw = 2.5, ls = '--')
    ax3.set_xlabel('Network size, ' + r'$N$', fontsize = 14)
    ax3.set_ylabel('Mean clustering \ncoefficient, ' + r'$\left \langle C \right \rangle$', fontsize = 14)
    #ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim(0.05, 1.5)


    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    x_d = df_net_feats.N.values
    y_d = df_net_feats.d_mean.values
    ax4.scatter(x_d, y_d, c='#175ac6', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.6, zorder=1)
    #x_d_range = list(range(10, max(x_d)))
    #barabasi_albert_range_d = [ (np.log(i) / np.log(np.log(i))) for i in x_d_range ]
    x_d_sort = list(set(x_d.tolist()))
    x_d_sort.sort()
    model_d = nt.distanceBarabasiAlbert(x_d, y_d)
    results_d = []
    for b0 in b0_start:
        for z in z_start:
            start_params_d = [b0, z]
            result_d = model_d.fit(start_params = start_params_d)
            results_d.append(result_d)
    AICs_d = [result_d.aic for result_d in results_d]
    best_d = results_d[AICs_d.index(min(AICs_d))]
    best_CI_FIC_d = nt.CI_FIC(best_d)
    best_CI_d = best_d.conf_int()
    best_d_params = best_d.params

    barabasi_albert_range_d_ll = nt.distance_BA(np.sort(x_d), best_d_params[0])
    ax4.plot(np.sort(x_C), barabasi_albert_range_d_ll, c = 'k', lw = 2.5,
        ls = '--', zorder = 2)
    #random_range = [ np.log(i) for i in x_d_range ]
    #ax4.plot(x_d_range, random_range, c = 'r', lw = 2.5, ls = '--')
    ax4.set_xlabel('Network size, ' + r'$N$', fontsize = 14)
    ax4.set_ylabel('Mean distance, ' + r'$\left \langle d \right \rangle$', fontsize = 14)
    #ax4.set_xscale('log')

    plt.tight_layout()
    fig_name = nt.get_path() + '/figs/fig6.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





####### network figs
def plot_edge_dist():
    df_path = nt.get_path() + '/data/Tenaillon_et_al/network.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    k_list = []
    for index, row in df.iterrows():
        k_row = sum(i >0 for i in row.values) - 1
        if k_row > 0:
            k_list.append(k_row)

    k_count = dict(Counter(k_list))
    k_count = {k: v / total for total in (sum(k_count.values()),) for k, v in k_count.items()}
    #x = np.log10(list(k_count.keys()))
    #y = np.log10(list(k_count.values()))
    k_mean = np.mean(k_list)
    print("mean k = " + str(k_mean))
    print("N = " + str(df.shape[0]))
    x = list(k_count.keys())
    y = list(k_count.values())

    x_poisson = list(range(1, 100))
    y_poisson = [(math.exp(-k_mean) * ( (k_mean ** k)  /  math.factorial(k) )) for k in x_poisson]

    fig = plt.figure()
    plt.scatter(x, y, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    plt.plot(x_poisson, y_poisson)
    plt.xlabel("Number of edges, k", fontsize = 16)
    plt.ylabel("Frequency", fontsize = 16)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.001, 1)

    fig.tight_layout()
    fig.savefig(nt.get_path() + '/figs/edge_dist.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def plot_cluster_dist():
    df_path = nt.get_path() + '/data/Tenaillon_et_al/network_CCs.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    k_count = dict(Counter(df.C_i.values))
    k_count = {k: v / total for total in (sum(k_count.values()),) for k, v in k_count.items()}
    #x = np.log10(list(k_count.keys()))
    #y = np.log10(list(k_count.values()))
    # cluster kde
    C_i = df.C_i.values
    grid_ = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.1, 10, 50)},
                    cv=20) # 20-fold cross-validation
    grid_.fit(C_i[:, None])
    x_grid_ = np.linspace(0, 2.5, 1000)
    kde_ = grid_.best_estimator_
    pdf_ = np.exp(kde_.score_samples(x_grid_[:, None]))
    pdf_ = [x / sum(pdf_) for x in pdf_]

    x = list(k_count.keys())
    y = list(k_count.values())

    #x_poisson = list(range(1, 100))
    #y_poisson = [(math.exp(-k_mean) * ( (k_mean ** k)  /  math.factorial(k) )) for k in x_poisson]

    fig = plt.figure()
    #plt.scatter(x, y, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    plt.plot(x_grid_, pdf_)
    plt.ylabel("Clustering coefficient, " + r'$C_{i}$', fontsize = 16)
    plt.xlabel("Number of edges, " + r'$k$', fontsize = 16)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(0, 1)
    fig.tight_layout()
    fig.savefig(nt.get_path() + '/figs/C_dist.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_nodes_over_time():
    directory = nt.get_path() + '/data/Good_et_al/networks_BIC/'
    time_nodes = []
    for filename in os.listdir(directory):
        df = pd.read_csv(directory + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        time_nodes.append((int(time), df.shape[0]))
    time_nodes_sorted = sorted(time_nodes, key=lambda tup: tup[0])
    x = [i[0] for i in time_nodes_sorted]
    y = [i[1] for i in time_nodes_sorted]

    x_pred = list(set(x))
    x_pred.sort()
    y_pred = [min(y) + x_pred_i + 1 for x_pred_i in list(range(len(x_pred) ))]


    fig = plt.figure()
    plt.scatter(x, y, marker = "o", edgecolors='#244162', c = '#175ac6', s = 120, zorder=3)
    plt.plot(x_pred, y_pred)
    plt.xlabel("Time (generations)", fontsize = 18)
    plt.ylabel('Network size, ' + r'$N$', fontsize = 18)
    plt.ylim(5, 500)
    plt.yscale('log')

    fig.tight_layout()
    fig.savefig(nt.get_path() + '/figs/good_N_vs_time.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_kmax_over_time():
    directory = nt.get_path() + '/data/Good_et_al/networks_BIC/'
    time_kmax = []
    for filename in os.listdir(directory):
        df = pd.read_csv(directory + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        time_kmax.append((int(time), max(df.astype(bool).sum(axis=0).values)))

    time_kmax_sorted = sorted(time_kmax, key=lambda tup: tup[0])
    x = [i[0] for i in time_kmax_sorted]
    y = [i[1] for i in time_kmax_sorted]
    x = np.log10(x)
    y = np.log10(y)

    #df_rndm_path = nt.get_path() + '/data/Good_et_al/networks_BIC_rndm.txt'
    #df_rndm = pd.read_csv(df_rndm_path, sep = '\t', header = 'infer')

    #x_rndm = np.log10(df_rndm.Generations.values)
    #y_rndm = np.log10(df_rndm.Generations.values)

    fig = plt.figure()
    #plt.scatter(x, y, marker = "o", edgecolors='none', c = '#175ac6', s = 120, zorder=3)
    plt.scatter(x, y, c='#175ac6', marker = 'o', s = 120, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.8, zorder=3)#, edgecolors='none')
    #plt.scatter(x_rndm, y_rndm, marker = "o", edgecolors='none', c = 'blue', s = 120, alpha = 0.1)
    '''using some code from ken locey, will cite later'''

    df = pd.DataFrame({'t': list(x)})
    df['kmax'] = list(y)
    f = smf.ols('kmax ~ t', df).fit()

    R2 = f.rsquared
    pval = f.pvalues
    intercept = f.params[0]
    slope = f.params[1]
    X = np.linspace(min(x), max(x), 1000)
    Y = f.predict(exog=dict(t=X))

    st, data, ss2 = summary_table(f, alpha=0.05)
    print(ss2)
    fittedvalues = data[:,2]
    pred_mean_se = data[:,3]
    pred_mean_ci_low, pred_mean_ci_upp = data[:,4:6].T
    pred_ci_low, pred_ci_upp = data[:,6:8].T

    slope_to_gamme = (1/slope) + 1

    plt.fill_between(x, pred_ci_low, pred_ci_upp, color='#175ac6', lw=0.5, alpha=0.2)
    #'$^\frac{1}{1 - '+str(round(slope_to_gamme,2))+'}$'
    #plt.text(2.4, 2.1, r'$k_{max}$'+ ' = '+str(round(10**intercept,2))+'*'+r'$t$'+ '$^{\frac{1}{1 - '+str(round(slope_to_gamme,2))+'}}$', fontsize=10, color='k', alpha=0.9)
    plt.text(2.4, 2.05, r'$k_{max}$'+ ' = ' + str(round(10**intercept,2)) + '*' + r'$t^ \frac{1}{\,' + str(round(slope_to_gamme,2)) + '- 1}$', fontsize=12, color='k', alpha=0.9)
    plt.text(2.4, 1.94,  r'$r^2$' + ' = ' +str("%.2f" % R2), fontsize=12, color='0.2')
    plt.plot(X.tolist(), Y.tolist(), '--', c='k', lw=2, alpha=0.8, color='k', label='Power-law')

    #plt.plot(t_x, t_y)
    plt.xlabel("Time (generations), " + r'$\mathrm{log}_{10}$', fontsize = 18)
    plt.ylabel(r'$k_{max}, \;  \mathrm{log}_{10}$', fontsize = 18)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(0.001, 1)

    fig.tight_layout()
    fig.savefig(nt.get_path() + '/figs/good_kmax_vs_time.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_cluster():
    df_path = nt.get_path() + '/data/Good_et_al/network_features.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer')

    fig = plt.figure()
    x = df.N.values
    y = df.C_mean.values
    #plt.scatter(x, y, marker = "o", edgecolors='none', c = '#87CEEB', s = 120, zorder=3)
    plt.scatter(x, y, c='#175ac6', marker = 'o', s = 120, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.9, zorder=3)

    x_range = list(range(10, max(x)))
    barabasi_albert_range = [ ((np.log(i) ** 2) / i) for i in x_range ]
    random_range = [(1/i) for i in x_range ]
    plt.plot(x_range, barabasi_albert_range, c = 'k', lw = 2.5, ls = '--')

    plt.xlabel('Network size, ' + r'$N$', fontsize = 18)
    plt.ylabel('Mean clustering coefficient, ' + r'$\left \langle C \right \rangle$', fontsize = 16)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.05, 1.5)

    fig.tight_layout()
    fig.savefig(nt.get_path() + '/figs/good_N_vs_C.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def plot_distance():
    df_path = nt.get_path() + '/data/Good_et_al/network_features.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer')

    fig = plt.figure()
    x = df.N.values
    y = df.d_mean.values
    #plt.scatter(x, y, marker = "o", edgecolors='none', c = '#87CEEB', s = 120, zorder=3)
    plt.scatter(x, y, c='#175ac6', marker = 'o', s = 120, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.9, zorder=3)

    x_range = list(range(10, max(x)))
    barabasi_albert_range = [ (np.log(i) / np.log(np.log(i))) for i in x_range ]
    random_range = [np.log(i) for i in x_range ]
    plt.plot(x_range, barabasi_albert_range, c = 'r', lw = 2.5, ls = '--')
    plt.plot(x_range, random_range, c = 'k', lw = 2.5, ls = '--')

    plt.xlabel('Network size, ' + r'$N$', fontsize = 18)
    plt.ylabel('Mean distance, ' + r'$\left \langle d \right \rangle$', fontsize = 16)
    #plt.xscale('log')
    #plt.ylim(0.05, 1.5)

    fig.tight_layout()
    fig.savefig(nt.get_path() + '/figs/good_N_vs_d.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_t_k_node():
    network_dir = nt.get_path() + '/data/Good_et_al/networks_naive/'
    node_dict = {}
    for filename in os.listdir(network_dir):
        df = pd.read_csv(network_dir + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        node_dict[time] = {}
        for index, row in df.iterrows():
            k_row = sum(i != 0 for i in row.values) - 1
            node_dict[time][index] = k_row

    node_df = pd.DataFrame.from_dict(node_dict)
    idx = node_df.sum(axis = 1).sort_values(ascending=False).head(10).index
    node_df_idx = node_df.ix[idx]
    #node_df_idx = node_df.ix[ ['malT'] ]
    colors = ['firebrick', 'darkorange', 'gold', 'darkgreen', 'palegreen', \
                'navy', 'royalblue', 'black', 'teal', 'dimgrey']
    color_count = 0

    fig = plt.figure()
    nodes = node_df.index.values
    for index, row in node_df_idx.iterrows():
        print(index)
        row = row.dropna()
        row_x = [ int(x) for x in row.index.values]
        row_y = row.values
        row_xy = list(zip(row_x, row_y))
        row_xy.sort(key=lambda tup: tup[1])  # sorts in place
        row_x_sort = [x[0] for x in row_xy]
        row_y_sort = [x[1] for x in row_xy]
        plt.scatter(row_x_sort, row_y_sort, c=colors[color_count], marker = 'o', s = 120, \
            edgecolors='k', linewidth = 0.6, alpha = 0.9)

        color_count += 1

    plt.xlabel('Time (generations)', fontsize = 18)
    plt.ylabel('k(t)', fontsize = 18)
    plt.xlim(2500, 60000)
    plt.ylim(1, 200)
    plt.xscale('log')
    plt.yscale('log')
    fig.tight_layout()
    fig.savefig(nt.get_path() + '/figs/good_t_vs_k_node.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_end_network():
    network_path = nt.get_path() + '/data/Good_et_al/networks_naive/network_62750.txt'
    df = pd.read_csv(network_path, sep = '\t', header = 'infer', index_col = 0)
    print(df)

#plot_t_k_node()
#fig6()
plot_end_network()
