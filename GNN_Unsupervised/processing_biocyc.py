# ### Imports
import os
import sys

dir = os.getcwd() + "/GNN_Unsupervised"
sys.path.append(dir)

from tqdm import tqdm
from utils.utils import *

from mpl_toolkits import mplot3d
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
# from utils.utils import *

# import hdbscan
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import sys

# %load_ext autotime

import json

def main (experiment):
    # ### Parameters
    # Opening JSON file

    file = open("{}/input/parameters_{}.json".format(dir, experiment.id))
    params = json.load(file)

    # raw_data_folder =  params["raw_folder"]
    # print("Raw folder:\t", raw_data_folder)

    exp = params["exp"]
    print("Exp:\t\t", exp)

    method = params["method"]
    print("Method:\t\t", method)

    groups_id = params["groups_id"]
    print("Group id:\t", groups_id)

    option = params["option"]
    print("Option:\t\t", option)

    control = params["control"]
    controls = control.split(",")
    print("Controls:\t", controls)

    for item1 in controls:
        for item2 in groups_id:
            if item1 != item2:
                groups = [item1, item2] # change
                print("Groups:\t\t", groups)

                # ### Load data
                # load dataset groups
                df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(dir, exp), index_col=0)        
                df_join_raw.index = df_join_raw.index.astype("str")

                # ### BioCyc
                # get filter graphs

                dict_graphs = {}

                for group in groups:
                    df_common_edges = pd.read_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}.csv".format(dir, exp, method, group, option),
                                                        dtype={"source": "string", "target": "string"})
                    G = nx.from_pandas_edgelist(df_common_edges, edge_attr=["weight"])
                    dict_graphs[group] = G

                # get nodes
                dict_nodes = {}
                for group in groups:
                    dict_nodes[group] = set(list(dict_graphs[group].nodes()))

                # set operation
                dict_set_operation = {}
                for group in groups:
                    dict_nodes_aux = dict_nodes.copy()
                    nodes_aux = dict_nodes_aux.pop(group)
                    unique_nodes = nodes_aux - set.union(*list(dict_nodes_aux.values()))

                    dict_set_operation[group] = unique_nodes

                dict_set_operation["-".join(groups)] = set.intersection(*list(dict_nodes.values()))

                # print set size
                for key, value in dict_set_operation.items():
                    print(key, len(value))

                # delete nodes without metabollities name
                for group in dict_set_operation:
                    inter = dict_set_operation[group] & set(list(df_join_raw.index.values))
                    dict_set_operation[group] = list(inter)

                # print set size
                for key, value in dict_set_operation.items():
                    print(key, len(value))

                # mapping metabolite name with ratio (2)
                for key, value in dict_set_operation.items():
                    nodes = dict_set_operation[key]

                    df_biocyc = pd.DataFrame()
                    df_biocyc["m/z"] = nodes

                    list_data = []
                    for group in groups:
                        df_aux = df_join_raw.filter(like=group)
                        df_aux = df_aux.loc[nodes]

                        # df_biocyc["mean-{}".format(group)] = df_aux.mean(axis=1).values
                        # df_biocyc["log-{}".format(group)] = np.log10(df_aux.mean(axis=1).values)
                        list_data.append(df_aux.mean(axis=1).values)

                    df_biocyc["before"] = np.log10(list_data[0])
                    df_biocyc["after"] = np.log10(list_data[1])
                    df_biocyc["ratio"] = np.log10(np.divide(list_data[1], list_data[0]))

                    # df_biocyc["metabolities"] = df_metadata.loc[common_nodes]["Metabolites - Approved by Nicola"].values
                    df_biocyc.insert(1, "metabolities", df_join_raw.loc[nodes]["Metabolite name"].values)
                    df_biocyc = df_biocyc.iloc[:, 1:]

                    # save
                    df_biocyc.to_csv("{}/output/{}/biocyc/biocyc_{}_{}_{}.csv".format(dir, exp, method, key, option), 
                                    index=False, header=False, sep="\t")
                    # df_biocyc.head()

                # mapping metabolite name with ratio (3)
                """ for key, value in dict_set_operation.items():
                    nodes = dict_set_operation[key]

                    df_biocyc = pd.DataFrame()
                    df_biocyc["m/z"] = nodes

                    for group in groups_id:
                        df_aux = df_join_raw.filter(like=group)
                        df_aux = df_aux.loc[nodes]

                        # df_biocyc["mean-{}".format(group)] = df_aux.mean(axis=1).values
                        df_biocyc["log-{}".format(group)] = np.log10(df_aux.mean(axis=1).values)

                    # df_biocyc["metabolities"] = df_metadata.loc[common_nodes]["Metabolites - Approved by Nicola"].values
                    df_biocyc.insert(1, "metabolities", df_metadata.loc[nodes]["Metabolites - Approved by Nicola"].values)

                    df_biocyc = df_biocyc.iloc[:, 1:]
                    print(key, df_biocyc.shape)
                    # save
                    df_biocyc.to_csv("{}/output/{}/biocyc/biocyc_{}_{}_{}.csv".format(dir, exp, methods[0], key, options[0]), 
                                    index=False, header=False, sep="\t")
                    # df_biocyc.head() """

                # df_biocyc = pd.read_csv("{}/output/{}/biocyc/biocyc_{}_{}_{}.csv".format(dir, exp, method, "-".join(groups), option), sep="\t")