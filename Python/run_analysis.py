from __future__ import division
import os, re
from collections import Counter
import numpy as np
import pandas as pd
import network_tools as nt



def get_likelihood_matrices():
    df_good_path = nt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
    df_good =  pd.read_csv(df_good_path, sep = '\t', header = 'infer', index_col = 0)
    df_good_delta = nt.likelihood_matrix(df_good, 'Good_et_al').get_likelihood_matrix()
    df_good_delta_out = nt.get_path() + '/data/Good_et_al/gene_by_pop_delta.txt'
    df_good_delta.to_csv(df_good_delta_out, sep = '\t', index = True)

    df_good_poly_path = nt.get_path() + '/data/Good_et_al/gene_by_pop_poly.txt'
    df_good_poly =  pd.read_csv(df_good_poly_path, sep = '\t', header = 'infer', index_col = 0)
    df_good_poly_delta = nt.likelihood_matrix(df_good_poly, 'Good_et_al').get_likelihood_matrix()
    df_good_poly_delta_out = nt.get_path() + '/data/Good_et_al/gene_by_pop_poly_delta.txt'
    df_good_poly_delta.to_csv(df_good_poly_delta_out, sep = '\t', index = True)



'''Network code'''

def get_naive_good_network():
    out_directory = nt.get_path() + '/data/Good_et_al/networks_naive'
    df_path = nt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_exclude = nt.complete_nonmutator_lines()
    to_exclude.append('p5')
    df_nonmut = df[df.index.str.contains('|'.join( to_exclude))]
    # remove columns with all zeros
    df_nonmut = df_nonmut.loc[:, (df_nonmut != 0).any(axis=0)]
    time_points = [ int(x.split('_')[1]) for x in df_nonmut.index.values]
    time_points_set = sorted(list(set([ int(x.split('_')[1]) for x in df_nonmut.index.values])))
    for time_point in time_points_set:
        print(time_point)
        df_time_point = df_nonmut[df_nonmut.index.to_series().str.contains('_' + str(time_point))]
        df_time_point = df_time_point.loc[:, (df_time_point != 0).any(axis=0)]
        network_tp = nt.reconstruct_naive_network(df_time_point)
        network_tp.to_csv(out_directory + '/network_' + str(time_point) + '.txt', sep = '\t', index = True)


def get_network_clustering_coefficients(dataset = 'good', kmax = True, reconstruct = 'naive'):
    if dataset == 'tenaillon':
        # df is a numpy matrix or pandas dataframe containing network interactions
        df_path = nt.get_path() + '/data/Tenaillon_et_al/network.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        if kmax == False:
            df = df.drop('kpsD', axis=0)
            df = df.drop('kpsD', axis=1)
            df_out = open(nt.get_path() + '/data/Tenaillon_et_al/network_CCs_no_kmax.txt', 'w')
        else:
            df_out = open(nt.get_path() + '/data/Tenaillon_et_al/network_CCs.txt', 'w')
        df_out.write('\t'.join(['Gene', 'k_i', 'C_i']) + '\n')
        for index, row in df.iterrows():
            k_row = sum(i != 0 for i in row.values) - 1
            if (k_row == 0) or (k_row == 1):
                C_i = 0
            else:
                non_zero = row.nonzero()
                row_non_zero = row[non_zero[0]]
                # drop the node
                row_non_zero = row_non_zero.drop(labels = [index])
                L_i = 0
                for index_gene, gene in row_non_zero.iteritems():
                    row_non_zero_list = row_non_zero.index.tolist()
                    row_non_zero_list.remove(index_gene)
                    df_subset = df.loc[[index_gene]][row_non_zero_list]
                    L_i += sum(sum(i != 0 for i in df_subset.values))
                # we don't multiply L_i by a factor of 2 bc we're double counting edges
                C_i =  L_i  / (k_row * (k_row-1) )
            df_out.write('\t'.join([index, str(k_row), str(C_i)]) + '\n')
        df_out.close()

    elif dataset == 'good':
        if reconstruct == 'naive':
            directory = nt.get_path() + '/data/Good_et_al/networks_naive/'
            df_out = open(nt.get_path() + '/data/Good_et_al/network_naive_CCs.txt', 'w')
        elif reconstruct == 'BIC':
            directory = nt.get_path() + '/data/Good_et_al/networks_BIC/'
            df_out = open(nt.get_path() + '/data/Good_et_al/network_BIC_CCs.txt', 'w')

        df_out.write('\t'.join(['Generations', 'Gene', 'k_i', 'C_i']) + '\n')
        for filename in os.listdir(directory):
            if filename == '.DS_Store':
                continue
            df = pd.read_csv(directory + filename, sep = '\t', header = 'infer', index_col = 0)
            gens = filename.split('.')
            time = re.split('[_.]', filename)[1]
            print(time)
            for index, row in df.iterrows():
                k_row = sum(i != 0 for i in row.values) - 1
                if (k_row == 0) or (k_row == 1):
                    C_i = float(0)
                else:
                    non_zero = row.nonzero()
                    row_non_zero = row[non_zero[0]]
                    # drop the node
                    row_non_zero = row_non_zero.drop(labels = [index])
                    L_i = 0
                    for index_gene, gene in row_non_zero.iteritems():
                        row_non_zero_list = row_non_zero.index.tolist()
                        row_non_zero_list.remove(index_gene)
                        df_subset = df.loc[[index_gene]][row_non_zero_list]
                        L_i += sum(sum(i != 0 for i in df_subset.values))
                    # we don't multiply L_i by a factor of 2 bc we're double counting edges
                    C_i =  L_i  / (k_row * (k_row-1) )
                df_out.write('\t'.join([str(time), index, str(k_row), str(C_i)]) + '\n')

        df_out.close()


def run_network_permutation_rndm(iter = 1000, include_kmax = True):
    df_path = nt.get_path() + '/data/Tenaillon_et_al/network.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df = df.drop('kpsD', axis=0)
    df = df.drop('kpsD', axis=1)
    df_out = open(nt.get_path() + '/data/Tenaillon_et_al/permute_network_rndm_nokmax.txt', 'w')
    df_out.write('\t'.join(['Iteration', 'k_max', 'k_mean', 'C_mean', 'C_mean_no1or2', 'd_mean']) + '\n')

    # get m using likelihood
    for i in range(iter):
        print("Iteration " + str(i))
        df_rndm = nt.get_random_network_edges(df)
        k_i_rndm = []
        C_i_rndm_list = []

        for index, row in df_rndm.iterrows():
            k_row = sum(i != 0 for i in row.values) - 1
            k_i_rndm.append(k_row)
            if (k_row == 0) or (k_row == 1):
                C_i_rndm_list.append(float(0))
            else:
                non_zero = row.nonzero()
                row_non_zero = row[non_zero[0]]
                # drop the node
                row_non_zero = row_non_zero.drop(labels = [index])
                L_i = 0
                for index_gene, gene in row_non_zero.iteritems():
                    row_non_zero_list = row_non_zero.index.tolist()
                    row_non_zero_list.remove(index_gene)
                    df_subset = df.loc[[index_gene]][row_non_zero_list]
                    L_i += sum(sum(i != 0 for i in df_subset.values))
                # we don't multiply L_i by a factor of 2 bc we're double counting edges
                C_i =  L_i  / (k_row * (k_row-1) )
                C_i_rndm_list.append(C_i)

        k_max = max(k_i_rndm)
        k_mean = np.mean(k_i_rndm)
        C_mean = np.mean(C_i_rndm_list)
        C_mean_no1or2 = np.mean([l for l in C_i_rndm_list if l > 0])
        distance_df = nt.networkx_distance(df_rndm)

        df_out.write('\t'.join([str(i), str(k_max), str(k_mean), str(C_mean), str(C_mean_no1or2), str(distance_df)]) + '\n')

    df_out.close()


def run_network_permutation_ba():
    df_path = nt.get_path() + '/data/Tenaillon_et_al/network.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_out = open(nt.get_path() + '/data/Tenaillon_et_al/permute_network_ba.txt', 'w')
    df_out.write('\t'.join(['Iteration', 'k_max', 'k_mean', 'C_mean', 'C_mean_no1or2', 'd_mean']) + '\n')

    k_list = []
    for index, row in df.iterrows():
        k_row = sum(i != 0 for i in row.values) - 1
        if k_row > 0:
            k_list.append(k_row)
    count_k_list = Counter(k_list)
    count_k_list_sum = sum(count_k_list.values())
    x = count_k_list.keys()
    y = [(i / count_k_list_sum) for i in count_k_list.values()]
    count_k_list.pop(max(x), None)
    x_no_max = list(count_k_list.keys())
    y_no_max = [(i / (count_k_list_sum-1)) for i in count_k_list.values()]


    model_no_max = nt.continuumBarabasiAlbert(x_no_max, y_no_max)
    m_start = [1, 2, 3, 4]
    z_start = [-2,-0.5]
    results = []
    for m in m_start:
        for z in z_start:
            start_params = [m, z]
            result = model_no_max.fit(start_params = start_params)
            results.append(result)
    AICs = [result.aic for result in results]
    best = results[AICs.index(min(AICs))]
    best_CI_FIC = nt.CI_FIC(best)
    best_CI = best.conf_int()
    best_params = best.params
    print(best_params)

    #barabasi_albert_range_C_ll = nt.cluster_BA(np.sort(x_C), best_params[0])



    df_out.close()






def get_good_network_features(reconstruct = 'naive'):
    if reconstruct == 'naive':
        directory = nt.get_path() + '/data/Good_et_al/networks_naive/'
        df_out = open(nt.get_path() + '/data/Good_et_al/network_naive_features.txt', 'w')
        df_clust_path = nt.get_path() + '/data/Good_et_al/network_naive_CCs.txt'
    elif reconstruct == 'BIC':
        directory = nt.get_path() + '/data/Good_et_al/networks_BIC/'
        df_out = open(nt.get_path() + '/data/Good_et_al/network_BIC_features.txt', 'w')
        df_clust_path = nt.get_path() + '/data/Good_et_al/network_BIC_CCs.txt'

    df_out_columns = ['Generations', 'N', 'k_max', 'k_mean', 'C_mean', 'C_mean_no1or2', 'd_mean']
    df_out.write('\t'.join(df_out_columns) + '\n')
    df_clust = pd.read_csv(df_clust_path, sep = '\t', header = 'infer')#, index_col = 0)
    for filename in os.listdir(directory):
        if filename == '.DS_Store':
            continue
        df = pd.read_csv(directory + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        df_clust_time = df_clust.loc[df_clust['Generations'] == int(time)]
        N = df.shape[0]
        k_max = max(df_clust_time.k_i.values)
        k_mean = np.mean(df_clust_time.k_i.values)
        C_mean = np.mean(df_clust_time.C_i.values)
        C_mean_no1or2 = np.mean(df_clust_time.loc[df_clust_time['k_i'] >= 2].C_i.values)

        distance_df = nt.networkx_distance(df)
        print(time)
        print(distance_df)

        row = [str(time), str(N), str(k_max), str(k_mean), str(C_mean), str(C_mean_no1or2), str(distance_df)]
        df_out.write('\t'.join(row) + '\n')

    df_out.close()


#get_network_clustering_coefficients()
get_naive_good_network()
#get_good_network_features()
