from __future__ import division
import os, pickle, math, random, itertools
from itertools import combinations
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import comb
from scipy.special import binom
from statsmodels.base.model import GenericLikelihoodModel
import networkx as nx
import scipy.stats as stats



def get_path():
    return os.path.expanduser("~/GitHub/NetworkEvolution")

def complete_nonmutator_lines():
    return ['m5','m6','p1','p2','p4','p5']

def nonmutator_shapes():
    return {'m5': 'o','m6':'s','p1':'^','p2':'D','p4':'P','p5':'X'}


def complete_mutator_lines():
    return ['m1','m4','p3']


class likelihood_matrix:
    def __init__(self, df, dataset):
        self.df = df
        self.dataset = dataset

    def get_gene_lengths(self, **keyword_parameters):
        if self.dataset == 'Good_et_al':
            conv_dict = cd.good_et_al().parse_convergence_matrix(get_path() + "/data/Good_et_al/gene_convergence_matrix.txt")
            length_dict = {}
            if ('gene_list' in keyword_parameters):
                for gene_name in keyword_parameters['gene_list']:
                    length_dict[gene_name] = conv_dict[gene_name]['length']
                #for gene_name, gene_data in conv_dict.items():
            else:
                for gene_name, gene_data in conv_dict.items():
                    length_dict[gene_name] = conv_dict[gene_name]['length']
            return(length_dict)

        elif self.dataset == 'Tenaillon_et_al':
            with open(get_path() + '/data/Tenaillon_et_al/gene_size_dict.txt', 'rb') as handle:
                length_dict = pickle.loads(handle.read())
                return(length_dict)

    def get_likelihood_matrix(self):
        #df_in = get_path() + '/data/' + self.dataset + '/gene_by_pop.txt'
        #df = pd.read_csv(df_in, sep = '\t', header = 'infer', index_col = 0)
        genes = self.df.columns.tolist()
        genes_lengths = self.get_gene_lengths(gene_list = genes)
        L_mean = np.mean(list(genes_lengths.values()))
        L_i = np.asarray(list(genes_lengths.values()))
        N_genes = len(genes)
        m_mean = self.df.sum(axis=1) / N_genes

        for index, row in self.df.iterrows():
            m_mean_j = m_mean[index]
            delta_j = row * np.log((row * (L_mean / L_i)) / m_mean_j)
            self.df.loc[index,:] = delta_j

        #out_name = get_path() + '/data/' + self.dataset + '/gene_by_pop_delta.txt'

        df_new = self.df.fillna(0)
        # remove colums with all zeros
        df_new.loc[:, (df_new != 0).any(axis=0)]
        # replace negative values with zero
        df_new[df_new < 0] = 0
        return df_new
        #df_new.to_csv(out_name, sep = '\t', index = True)


# function to generate confidence intervals based on Fisher Information criteria
def CI_FIC(results):
    # standard errors = square root of the diagnol of a variance-covariance matrix
    ses = np.sqrt(np.absolute(np.diagonal(results.cov_params())))
    cfs = results.params
    lw = cfs - (1.96*ses)
    up = cfs +(1.96*ses)
    return (lw, up)



def reconstruct_naive_network(df):
    columns_list = df.columns.values.tolist()
    df_ = pd.DataFrame(index=columns_list, columns=columns_list)
    df_ = df_.fillna(0)
    for gene in columns_list:
        df_[gene][gene] = 1
    #for i, gene_i in enumerate(columns_list):
    #    for j, gene_j in enumerate(columns_list):
    #        if i > j:
    #            pairwise_dict[ gene_i + '-' + gene_j] = 0
    for index, row in df.iterrows():
        row_filter = row[row!=0]
        for i, row_gene_i in enumerate(row_filter.index.tolist()):
            for j, row_gene_j in enumerate(row_filter.index.tolist()):
                if i > j:
                    df_[row_gene_i][row_gene_j] += 1
                    df_[row_gene_j][row_gene_i] += 1

    return df_







def cluster_BA(N, b0):
    #N = np.sort(N)
    return b0 * ((np.log(N) ** 2) / N)

def distance_BA(N, b0):
    return b0 * (np.log(N) /  np.log( np.log(N) )   )


class clusterBarabasiAlbert(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(clusterBarabasiAlbert, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        b0 = params[0]
        z = params[1]
        # probability density function (pdf) is the same as dnorm
        exog_pred = cluster_BA(self.endog, b0 = b0)
        # need to flatten the exogenous variable
        LL = -stats.norm.logpdf(self.exog.flatten(), loc=exog_pred, scale=np.exp(z))
        return LL

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        if start_params is None:
            b0_start = 1
            z_start = 0.8

            start_params = np.asarray([b0_start, z_start])

        return super(clusterBarabasiAlbert, self).fit(start_params=start_params,
                                maxiter=maxiter, method = method, maxfun=maxfun,
                                **kwds)


class distanceBarabasiAlbert(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(distanceBarabasiAlbert, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        b0 = params[0]
        z = params[1]
        # probability density function (pdf) is the same as dnorm
        exog_pred = distance_BA(self.endog, b0 = b0)
        # need to flatten the exogenous variable
        LL = -stats.norm.logpdf(self.exog.flatten(), loc=exog_pred, scale=np.exp(z))
        return LL

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        if start_params is None:
            b0_start = 1
            z_start = 0.8

            start_params = np.asarray([b0_start, z_start])

        return super(distanceBarabasiAlbert, self).fit(start_params=start_params,
                                maxiter=maxiter, method = method, maxfun=maxfun,
                                **kwds)


def continuum_BA(k, m):
    # large k limit
    #return 2 * (m**2) *  (1 / (k**3))
    # exact equation
    return (2 * m * (m+1)) / (k * (k+1) * (k+2) )


class continuumBarabasiAlbert(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(continuumBarabasiAlbert, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        m = params[0]
        z = params[1]
        # probability density function (pdf) is the same as dnorm
        exog_pred = continuum_BA(self.endog, m = m)
        # need to flatten the exogenous variable
        LL = -stats.norm.logpdf(self.exog.flatten(), loc=exog_pred, scale=np.exp(z))
        return LL

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        if start_params is None:
            m_start = 1
            z_start = 0.8

            start_params = np.asarray([m_start, z_start])

        return super(continuumBarabasiAlbert, self).fit(start_params=start_params,
                                maxiter=maxiter, method = method, maxfun=maxfun,
                                **kwds)



def get_random_network_probability(df):
    # df is a numpy matrix or pandas dataframe containing network interactions
    N = df.shape[0]
    M = 0
    k_list = []
    for index, row in df.iterrows():
        # != 0 bc there's positive and negative interactions
        k_row = sum(i != 0 for i in row.values) - 1
        if k_row > 0:
            k_list.append(k_row)
    # run following algorithm to generate random network
    # (1) Start with N isolated nodes.
    # (2) Select a node pair and generate a random number between 0 and 1.
    #     If the number exceeds p, connect the selected node pair with a link,
    #     otherwise leave them disconnected.
    # (3) Repeat step (2) for each of the N(N-1)/2 node pairs.
    M = sum(k_list)
    p = M / binom(N, 2)
    matrix = np.ones((N,N))
    #node_pairs = list(combinations(range(int((N * (N-1)) / 2)), 2))
    node_pairs = list(combinations(range(N), 2))
    for node_pair in node_pairs:
        p_node_pair = random.uniform(0, 1)
        if p_node_pair > p:
            continue
        else:
            matrix[node_pair[0], node_pair[1]] = 0
            matrix[node_pair[1], node_pair[0]] = 0

    return matrix

def get_random_network_edges(df):
    nodes = df.index.tolist()
    edges = []
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i < j:
                edges.append(node_i + '-' + node_j)
    L = 0
    for index, row in df.iterrows():
        #k_row = sum(float(k) != float(0) for k in row.values) - 1
        k_row = len([i for i in row.values if i != 0]) - 1
        L += k_row

    new_edges = np.random.choice(np.asarray(edges), size=int(L/2), replace=False)
    new_edges_split = [x.split('-') for x in new_edges]
    matrix = pd.DataFrame(0, index= nodes, columns=nodes)
    for node in nodes:
        matrix.loc[node, node] = 1
    for new_edge in new_edges_split:
        matrix.loc[new_edge[0], new_edge[1]] = 1
        matrix.loc[new_edge[1], new_edge[0]] = 1

    L_test = 0
    for index_m, row_m in matrix.iterrows():
        k_row_m = len([i for i in row_m.values if i != 0]) - 1
        L_test += k_row_m

    return matrix


def networkx_distance(df):
    def get_edges(nodes):
        edges = []
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i < j:
                    pair_value = df.loc[node_i][node_j]
                    if pair_value > 0:
                        edges.append((node_i,node_j))
        return(edges)

    edges_full = get_edges(df.index.tolist())
    graph = nx.Graph()
    graph.add_edges_from(edges_full)

    if nx.is_connected(graph) == True:
        return(nx.average_shortest_path_length(graph))
    else:
        components = [list(x) for x in nx.connected_components(graph)]
        component_distances = []
        # return a grand mean if there are seperate graphs
        for component in components:
            component_edges = get_edges(component)
            graph_component = nx.Graph()
            graph_component.add_edges_from(component_edges)
            component_distance = nx.average_shortest_path_length(graph_component)
            component_distances.append(component_distance)
        return( np.mean(component_distances) )
