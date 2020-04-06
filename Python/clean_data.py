from __future__ import division
import os, pickle
from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd
import network_tools as nt

#nt.get_path()

class good_et_al:

    def __init__(self):
        self.populations = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', \
                        'p1', 'p2', 'p3', 'p4', 'p5', 'p6']

    def parse_convergence_matrix(self, filename):

        convergence_matrix = {}

        convergence_matrix_file = open(filename,"r")
        # Header line
        line = convergence_matrix_file.readline()
        populations = [item.strip() for item in line.split(",")[2:]]

        for line in convergence_matrix_file:

            items = line.split(",")
            gene_name = items[0].strip()
            length = float(items[1])

            convergence_matrix[gene_name] = {'length':length, 'mutations': {population: [] for population in populations}}

            for population, item in zip(populations,items[2:]):
                if item.strip()=="":
                    continue

                subitems = item.split(";")
                for subitem in subitems:
                    subsubitems = subitem.split(":")
                    mutation = (float(subsubitems[0]), float(subsubitems[1]), float(subsubitems[2]), float(subsubitems[3]))
                    convergence_matrix[gene_name]['mutations'][population].append(mutation)


        return convergence_matrix


    def reformat_convergence_matrix(self, mut_type = 'F'):
        conv_dict = self.parse_convergence_matrix(nt.get_path() + "data/Good_et_al/gene_convergence_matrix.txt")
        time_points = []
        new_dict = {}
        for gene_name, gene_data in conv_dict.items():
            for pop_name, mutations in gene_data['mutations'].items():
                for mutation in mutations:
                    time = int(mutation[0])
                    time_points.append(time)
        time_points = sorted(list(set(time_points)))
        for gene_name, gene_data in conv_dict.items():
            if gene_name not in new_dict:
                new_dict[gene_name] = {}
            for pop_name, mutations in gene_data['mutations'].items():
                if len(mutations) == 0:
                    continue

                mutations.sort(key=lambda tup: tup[0])
                # keep only fixed mutations
                #{'A':0,'E':1,'F':2,'P':3}
                if mut_type == 'F':
                    mutations = [x for x in mutations if int(x[1]) == 2]
                elif mut_type == 'P':
                    mutations = [x for x in mutations if (int(x[1]) == 3) ]#or (int(x[1]) == 0)]
                else:
                    print("Argument mut_type not recognized")

                if len(mutations) == 0:
                    continue
                for mutation in mutations:
                    if mut_type == 'F':
                        time = mutation[0]
                        remaining_time_points = time_points[time_points.index(time):]
                        for time_point in remaining_time_points:
                            pop_time = pop_name +'_' + str(int(time_point))
                            if pop_time not in new_dict[gene_name]:
                                new_dict[gene_name][pop_time] = 1
                            else:
                                new_dict[gene_name][pop_time] += 1
                    elif mut_type == 'P':
                        pop_time = pop_name +'_' + str(int(mutation[0]))
                        if pop_time not in new_dict[gene_name]:
                            new_dict[gene_name][pop_time] = 1
                        else:
                            new_dict[gene_name][pop_time] += 1

        df = pd.DataFrame.from_dict(new_dict)
        df = df.fillna(0)
        df = df.loc[:, (df != 0).any(axis=0)]
        if mut_type == 'F':
            df_out = nt.get_path() + 'data/Good_et_al/gene_by_pop.txt'
            #df_delta_out = mydir + 'data/Good_et_al/gene_by_pop_delta.txt'
        elif mut_type == 'P':
            df_out = nt.get_path() + 'data/Good_et_al/gene_by_pop_poly.txt'
            #df_delta_out = mydir + 'data/Good_et_al/gene_by_pop_poly_delta.txt'
        else:
            print("Argument mut_type not recognized")
        df.to_csv(df_out, sep = '\t', index = True)




class tenaillon_et_al:

    def clean_tenaillon_et_al(self):
        df_in = nt.get_path() + 'data/Tenaillon_et_al/1212986tableS2.csv'
        df_out = open(nt.get_path() + 'data/Tenaillon_et_al/1212986tableS2_clean.csv', 'w')
        category_dict = {}
        header = ['Lines', 'Position', 'Type', 'Change', 'Genic_status', 'Gene_nb', \
                    'Gene_name', 'Effect', 'Site_affected', 'Length', \
                    'Genic_type', 'Gene_nb_type', 'Gene_name_type', \
                    'Effect_type', 'Site_affected_type', 'Length_type']
        df_out.write(','.join(header) + '\n')
        # For genic, check whether genic + '_' + 7th column value in dict, if not, select
        # 'Genic' as key
        head_type = { 'Genic': ['Genic', 'Gene_nb', 'Gene_Name', 'Effect', 'codon_affected', 'gene_length_in_codon'] , \
                'Genic_Large_Deletion': ['Genic', 'Gene_nb', 'Gene_Name', 'Large_Deletion', 'bp_deleted_in_Gene', 'gene_length_bp' ], \
                'Genic_RNA':  ['Genic', 'Gene_nb', 'Gene_Name', 'RNA', 'bp_affected', 'gene_length_bp']  ,\
                'Intergenic_Intergenic': ['Intergenic', 'Previous_Gene_nb', 'Previous_Gene_Name_distance_bp', 'Effect', 'Next_Gene_Name_distance_bp', 'Intergenic_type'], \
                'Multigenic_Multigenic': ['Multigenic', 'First_Gene_nb', 'First_Gene_Name', 'Effect', 'Last_Gene_nb', 'Last_Gene_Name']}
        for i, line in enumerate(open(df_in, 'r')):
            line = line.strip().split(',')
            if (len(line) == 0) or (i in range(0, 5)) or (len(line[0]) == 0):
                continue
            else:
                line_type = line[4] + '_' + line[7]
                if line_type in head_type:
                    line_new = line + head_type[line_type]
                else:
                    line_new = line + head_type['Genic']
                df_out.write(','.join(line_new) + '\n')
        df_out.close()

    def pop_by_gene_tenaillon(self):
        pop_by_gene_dict = {}
        gene_size_dict = {}
        df_in = nt.get_path() + 'data/Tenaillon_et_al/1212986tableS2_clean.csv'
        for i, line in enumerate(open(df_in, 'r')):
            line_split = line.strip().split(',')
            if (line_split[4] == 'Intergenic') or \
            (i == 0) or \
            (line_split[9].isdigit() == False):
                continue
            gene_length_units = line_split[-1]
            gene_name = line_split[6]
            pop_name = line_split[0]
            if gene_length_units == 'gene_length_in_codon':
                gene_length = int(line_split[9]) * 3
            elif gene_length_units == 'gene_length_bp':
                gene_length = int(line_split[9])
            if gene_name not in gene_size_dict:
                gene_size_dict[gene_name] = gene_length

            if gene_name not in pop_by_gene_dict:
                pop_by_gene_dict[gene_name] = {}

            if pop_name not in pop_by_gene_dict[gene_name]:
                pop_by_gene_dict[gene_name][pop_name] = 1
            else:
                pop_by_gene_dict[gene_name][pop_name] += 1

        df = pd.DataFrame.from_dict(pop_by_gene_dict)
        df = df.fillna(0)
        # remove rows and columns with all zeros
        #df = df.loc[(df.sum(axis=1) != 0), (df.sum(axis=0) != 0)]
        df_out = nt.get_path() + 'data/Tenaillon_et_al/gene_by_pop.txt'
        df.to_csv(df_out, sep = '\t', index = True)
        gene_size_dict_out = nt.get_path() + 'data/Tenaillon_et_al/gene_size_dict.txt'
        with open(gene_size_dict_out, 'wb') as handle:
            pickle.dump(gene_size_dict, handle)


good_et_al().reformat_convergence_matrix(mut_type = 'P')
#good_et_al().reformat_convergence_matrix(mut_type = 'F')
