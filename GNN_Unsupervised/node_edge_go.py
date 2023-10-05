#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[19]:



from pyod.models.ecod import ECOD
from mpl_toolkits import mplot3d
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
# from utils.utils import *

# import hdbscan
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pingouin as pg
import sys


# DGI
# import argparse, time

# import dgl
import networkx as nx
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dgi.dgi import Classifier, DGI
# from dgl import DGLGraph
# from dgl.data import load_data, register_data_args, DGLDataset

import os

from tqdm import tqdm
import pandas as pd

import sys
# sys.path.append("../")

from utils.utils_go import *
from dgi.utils_dgi import *
# from vgae.utils_vgae import *

os.environ["DGLBACKEND"] = "pytorch"

# ### Parameters

# In[20]:


import json

dir = os.getcwd() + "/GNN_Unsupervised"
print(dir)

def main(experiment):
    # file = open("exp.json")
    # experiment = json.load(file)
    exp_num = str(experiment.id) # experiment["exp"]

    file = open("{}/output/{}/parameters.json".format(dir, exp_num))
    params = json.load(file)

    exp = params["exp"]
    print("Exp:\t\t", exp)

    method = params["method"]
    print("Method:\t\t", method)

    methods = params["methods"]
    print("Methods:\t", methods)

    dimension = params["dimension"]
    print("Dimension:\t", dimension)

    groups_id = params["groups_id"]
    print("Groups id:\t", groups_id)

    subgroups_id_ori = params["subgroups_id"]
    # subgroups_id = params["subgroups_id"]
    print("Subgroups id:\t", subgroups_id_ori)

    option = params["option"]
    print("Option:\t\t", option)

    options = params["options"]
    print("Options:\t", options)

    threshold = params["threshold"]
    print("Threshold:\t", threshold)

    seeds = params["seeds"]
    print("Seeds:\t\t", seeds)

    iterations = params["iterations"]
    print("Iterations:\t", iterations)

    """ if option:
        subgroups_id_op = {}
        for group in groups_id:
            subgroups_id_op[group] = [option]
    else:
        subgroups_id_op = subgroups_id
    print("Subgroups id op:", subgroups_id_op) """


    # ### Node embeddings

    # #### DGI

    # In[21]:


    # In[22]:


    # read raw data
    df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(dir, exp), index_col=0)
    # df_join_raw.index = df_join_raw.index.astype("str")
    df_join_raw = df_join_raw.iloc[:, 2:]
    df_join_raw

    # log10
    df_join_raw_log = log10_global(df_join_raw)
    df_join_raw_log.head()

    # node-embeddings + edge-embeddings
    for method in methods:
        for option in options:        
            for iteration in range(iterations):
                # ---
                # Node embeddings
                # ---
                subgroups_id = subgroups_id_ori.copy()
                
                torch.manual_seed(seeds[iteration])
                np.random.seed(seeds[iteration])
                
                if option:
                    for group in groups_id:
                        subgroups_id[group] = [option]
                
                print("Subgroups id:\t", subgroups_id)
                
                for group in tqdm(groups_id):
                    for subgroup in tqdm(subgroups_id[group]):
                        nodes_data = pd.read_csv("{}/output/{}/preprocessing/graphs_data/nodes_data_{}_{}.csv".format(dir, exp, group, subgroup)).iloc[:, 2:]
                        edges_data = pd.read_csv("{}/output/{}/preprocessing/graphs_data/edges_data_{}_{}.csv".format(dir, exp, group, subgroup))

                        if method == "dgi":
                            data = CustomDatasetDGI("g_{}_{}".format(group, subgroup), nodes_data, edges_data)
                            graph = data[0]

                            # train
                            args_ = args_dgi(dimension)
                            train_dgi(exp, graph, args_, method, group, subgroup, iteration)
                        else:
                            data = CustomDatasetVGAE("g_{}_{}".format(group, subgroup), nodes_data, edges_data)
                            graph = data[0]

                            # train
                            args_ = args_vgae(dimension)
                            train_vgae(exp, graph, args_, method, group, subgroup, iteration)
                    
                # ---
                # Edge embeddings
                # ---
                subgroups_id = subgroups_id_ori.copy()
                print(method, option)
                
                if option:
                    subgroups_id_op = {}
                    for group in groups_id:
                        subgroups_id_op[group] = [option]
                else:
                    subgroups_id_op = subgroups_id
                print("Subgroups id op:", subgroups_id_op)
                
                edge_embeddings_global(exp, method, groups_id, subgroups_id_op, iteration)
                
                for group in tqdm(groups_id):
                    df_edge_embeddings_concat = pd.DataFrame()
                    k = 0
                    for subgroup in tqdm(subgroups_id_op[group]):
                        df_edge_embeddings = pd.read_csv("{}/output/{}/edge_embeddings/edge-embeddings_{}_{}_{}_{}.csv".format(dir, exp, method, group, subgroup, iteration))
                        df_edge_embeddings["subgroup"] = [k] * len(df_edge_embeddings)
                        df_edge_embeddings_concat = pd.concat([df_edge_embeddings_concat, df_edge_embeddings])
                        k += 1
                    
                    df_edge_embeddings_concat.to_csv("{}/output/{}/edge_embeddings/edge-embeddings_concat_{}_{}_{}_{}.csv".format(dir, exp, method, group, option, iteration), index=False)
                        
                # outlier detection (ECOD)
                # dict_df_edge_embeddings_concat_outlier = {}
                dict_df_edge_embeddings_concat_filter = {}

                for group in tqdm(groups_id):
                    df_edge_embeddings_concat = pd.read_csv("{}/output/{}/edge_embeddings/edge-embeddings_concat_{}_{}_{}_{}.csv".format(dir, exp, method, group, option, iteration))

                    X_train = df_edge_embeddings_concat.iloc[:, 2:-1]

                    clf = ECOD()
                    clf.fit(X_train)

                    X_train["labels"] = clf.labels_ # binary labels (0: inliers, 1: outliers)

                    df_edge_embeddings_concat_filter = df_edge_embeddings_concat.copy()
                    df_edge_embeddings_concat_filter["labels"] = clf.labels_

                    # save
                    df_edge_embeddings_concat_filter.to_csv("{}/output/{}/edge_embeddings/edge-embeddings_concat_outlier_{}_{}_{}_{}.csv".format(dir, exp, method, group, option, iteration), index=False)
                    
                    df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter[df_edge_embeddings_concat_filter["labels"] == 0]
                    df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter.iloc[:, :-1]

                    # dict_df_edge_embeddings_concat_outlier[group] = X_train
                    dict_df_edge_embeddings_concat_filter[group] = df_edge_embeddings_concat_filter
                    
                # mapping idx with id
                for group in tqdm(groups_id):
                    df_aux = pd.DataFrame(())
                    k = 0
                    for subgroup in subgroups_id_op[group]:
                        df_nodes = pd.read_csv("{}/output/{}/preprocessing/graphs_data/nodes_data_{}_{}.csv".format(dir, exp, group, subgroup))
                        dict_id = dict(zip(df_nodes["idx"], df_nodes["id"]))

                        # mapping
                        df_edge_embeddings_concat_filter = dict_df_edge_embeddings_concat_filter[group]
                        df_edge_embeddings_concat_filter_aux = df_edge_embeddings_concat_filter[df_edge_embeddings_concat_filter["subgroup"] == k]
                        
                        # print(df_edge_embeddings_concat_filter)
                        df_edge_embeddings_concat_filter_aux["source"] = df_edge_embeddings_concat_filter_aux["source"].map(dict_id)
                        df_edge_embeddings_concat_filter_aux["target"] = df_edge_embeddings_concat_filter_aux["target"].map(dict_id)
                        df_aux = pd.concat([df_aux, df_edge_embeddings_concat_filter_aux])
                        k += 1
                    dict_df_edge_embeddings_concat_filter[group] = df_aux
                    
                # format id
                if option:
                    for group in tqdm(groups_id):
                        # format
                        df_edge_embeddings_concat_filter = dict_df_edge_embeddings_concat_filter[group]
                        df_edge_embeddings_concat_filter["source"] = df_edge_embeddings_concat_filter["source"].map(lambda x: int(x[1:]))
                        df_edge_embeddings_concat_filter["target"] = df_edge_embeddings_concat_filter["target"].map(lambda x: int(x[1:]))
                            
                # filter by different edges
                if option:
                    for group in tqdm(groups_id):
                        df_edge_embeddings_concat_filter = dict_df_edge_embeddings_concat_filter[group]
                        df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter[df_edge_embeddings_concat_filter["source"] != df_edge_embeddings_concat_filter["target"]]
                        dict_df_edge_embeddings_concat_filter[group] = df_edge_embeddings_concat_filter
                        
                # count edges and filter by count
                dict_df_edges_filter = {}
                for group in tqdm(groups_id):
                    # read
                    df_edge_embeddings_concat_filter = dict_df_edge_embeddings_concat_filter[group]
                    
                    # sort edges
                    sort_df_edges(df_edge_embeddings_concat_filter)

                    df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter[["source", "target"]].value_counts().to_frame()
                    df_edge_embeddings_concat_filter.reset_index(inplace=True)
                    df_edge_embeddings_concat_filter.columns = ["source", "target", "count"]
                    
                    # filter
                    df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter[df_edge_embeddings_concat_filter["count"] == len(subgroups_id[group])]
                    df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter.iloc[:, [0, 1]]
                    dict_df_edges_filter[group] = df_edge_embeddings_concat_filter
                    
                    df_edge_embeddings_concat_filter.sort_values(["source", "target"], ascending=True, inplace=True)
                    df_edge_embeddings_concat_filter.to_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}_{}.csv".format(dir, exp, method, group, option, iteration), index=False)
        


    # In[23]:


    # join
    list_details = []

    for method in methods:
        print(method)    
        for k, group in enumerate(groups_id):
            print(group)
            
            dict_df_edges_filter = {}
            dict_df_corr = {}
            dict_df_edges_filter_weight = {}
        
            for option in options:
                list_common_subgraph = []
                for iteration in range(iterations):
                    print("Option: ", option)
                    df_edges_filter_weight_filter = pd.read_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}_{}.csv".format(dir, exp, method, group, option, iteration))
                    # print(df_edges_filter_weight_filter)

                    G = nx.from_pandas_edgelist(df_edges_filter_weight_filter) # , edge_attr=["weight"])
                    # SG = G.subgraph([0, 1, 2, 3, 4, 5])
                    # graph_partial_detail(G, edges=True)
                    list_common_subgraph.append(G)
                    
                print("Union")
                # union
                U = nx.compose_all(list_common_subgraph)
                
                df_compose_subgraph = nx.to_pandas_edgelist(U)       
                dict_df_edges_filter[group] = df_compose_subgraph.iloc[:, [0, 1]]
                
                # correlation
                nodes = list(U.nodes())
                
                df_join_raw_filter = df_join_raw_log.loc[nodes, :]
                # df_join_raw_filter = df_join_raw_filter.filter(regex=group, axis=1)
                df_join_raw_filter = df_join_raw_filter.filter(like=group, axis=1)

                df_join_raw_filter_t= df_join_raw_filter.T
                # df_join_raw_filter_corr = df_join_raw_filter_t.corr(method="pearson")
                df_join_raw_filter_corr = pg.pcorr(df_join_raw_filter_t)
                dict_df_corr[group] = df_join_raw_filter_corr
                
                # get new correlation
                df_edges_filter_weight = dict_df_edges_filter[group].copy()
                df_corr = dict_df_corr[group]

                df_edges_filter_weight["weight"] = df_edges_filter_weight.apply(lambda x: df_corr.loc[x["source"], x["target"]], axis=1)
                df_edges_filter_weight.sort_values(["source", "target"], ascending=True, inplace=True)
                dict_df_edges_filter_weight[group] = df_edges_filter_weight
                
                # details
                df_edges_filter_weight = dict_df_edges_filter_weight[group]
                G = nx.from_pandas_edgelist(df_edges_filter_weight, "source", "target", edge_attr="weight")
                # print(group, G.number_of_nodes(), G.number_of_edges())
                print("Before")
                print(method, group, option)
                # graph_partial_detail(G, edges=True)
                    
                # filter by abs(weight) >= threshold
                df_edges_filter_weight = dict_df_edges_filter_weight[group]
                df_edges_filter_weight_filter = df_edges_filter_weight[df_edges_filter_weight["weight"].abs() >= threshold]
                df_edges_filter_weight_filter.to_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}.csv".format(dir, exp, method, group, option), index=False)
                
                # print(group, U.edges(), 12)
                # graph_partial_detail(U, edges=True)
                
                # save
                # df_common_subgraph = nx.to_pandas_edgelist(U)
                # df_common_subgraph.to_csv("output/{}/common_edges/common_edges_union_{}_{}.csv".format(dir, exp, "-".join(methods), group), index=False)
                
                # common subgraph
                df_edges_filter_weight_filter = pd.read_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}.csv".format(dir, exp, method, group, option))
                df_edges_filter_weight_filter

                G = nx.from_pandas_edgelist(df_edges_filter_weight_filter, "source", "target", edge_attr="weight")
                # print(group, G.number_of_nodes(), G.number_of_edges())
                
                print("After")
                print(method, group, option)
                graph_partial_detail(G, edges=True)
                print("---")
                
                SG = G.subgraph([0, 1, 2, 3, 4, 5])
                list_details.append([method, group, option, SG.number_of_nodes(), SG.number_of_edges()])


    # In[24]:


    # summary
    df_comparation = pd.DataFrame(list_details, columns=["Method", "Group", "Option", "Num. nodes", "Num. edges"])
    df_comparation


    # In[25]:


    df_comparation_temp = df_comparation.set_index(["Method", "Group", "Option"])
    df_comparation_temp.sort_values(by=["Method", "Group", "Num. edges"], ascending=False, inplace=True)
    ax = df_comparation_temp.plot.bar(rot=90)
    ax.grid()

