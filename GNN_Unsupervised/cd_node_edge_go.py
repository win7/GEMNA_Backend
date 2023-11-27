#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[16]:


import json

from pyod.models.ecod import ECOD
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import pingouin as pg
import torch
import torch_geometric.transforms as T

from GNN_Unsupervised.utils.utils_go import *
# from dgi.utils_dgi import *
# from vgae.utils_vgae import *
# from vgae.utils_vgae_tg import *
from unsupervised_models.models import *

# os.environ["DGLBACKEND"] = "pytorch"
# %load_ext autotime


# In[17]:


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[18]:


import torch

# get_ipython().system('python -c "import torch; print(torch.version.cuda)"')
# get_ipython().system('python -c "import torch; print(torch.__version__)"')

dir = os.getcwd() + "/GNN_Unsupervised"
print(dir)

def main(experiment):
    # ### Parameters

    # In[19]:


    """ file = open("exp.json")
    experiment = json.load(file)
    exp_num = experiment["exp"] """
    exp_num = str(experiment.id)

    file = open("{}/output/{}/parameters.json".format(dir, exp_num))
    params = json.load(file)

    exp = params["exp"]
    print("Exp:\t\t", exp)

    methods = params["methods"]
    print("Methods:\t", methods)

    data_variations = params["data_variations"]
    print("Data variations:", data_variations)

    dimension = params["dimension"]
    print("Dimension:\t", dimension)

    threshold_corr = params["threshold_corr"]
    print("Threshold corr:\t", threshold_corr)

    iterations = params["iterations"]
    print("Iterations:\t", iterations)

    groups_id = params["groups_id"]
    print("Groups id:\t", groups_id)

    subgroups_id_ = params["subgroups_id"]
    print("Subgroups id:\t", subgroups_id_)

    seeds = params["seeds"]
    print("Seeds:\t\t", seeds)


    # ### Node-Edge embeddings

    # In[20]:


    # read raw data
    df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(dir, exp), index_col=0)
    df_join_raw = df_join_raw.iloc[:, 2:]
    df_join_raw

    # log10
    df_join_raw_log = log10_global(df_join_raw)
    df_join_raw_log.head()

    epochs = 50 # change
    cuda = 3    # change
    device = torch.device('cuda:{}'.format(cuda) if torch.cuda.is_available() else 'cpu')
    print(device)

    # node-embeddings + edge-embeddings
    for method in methods: # change
        for data_variation in data_variations: # change   
            for iteration in range(iterations):
                # ---
                # Node embeddings
                # ---
                subgroups_id = subgroups_id_.copy()
                
                torch.manual_seed(seeds[iteration])
                np.random.seed(seeds[iteration])
                
                if data_variation != "none":
                    for group in groups_id:
                        subgroups_id[group] = [data_variation]
                print("Subgroups id:\t", subgroups_id)
                
                for group in groups_id:
                    for subgroup in subgroups_id[group]:
                        nodes_data = pd.read_csv("{}/output/{}/preprocessing/graphs_data/nodes_data_{}_{}.csv".format(dir, exp, group, subgroup)).iloc[:, 2:]
                        edges_data = pd.read_csv("{}/output/{}/preprocessing/graphs_data/edges_data_{}_{}.csv".format(dir, exp, group, subgroup))

                        if method == "dgi":
                            data = CustomDatasetDGI("g_{}_{}".format(group, subgroup), nodes_data, edges_data)
                            graph = data[0]
                            
                            args_ = args_dgi(dimension)
                            train_dgi(exp, graph, args_, method, group, subgroup, iteration)
                        
                        elif method == "dgi-tran":
                            transform = T.Compose([
                                # T.NormalizeFeatures(), #
                                T.ToDevice(device),
                                # T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True, add_negative_train_samples=False),
                            ])
                            dataset = CustomDataset(nodes_data, edges_data, transform=transform)
                            model = DGI_Transductive(dataset, dimension, device)
                            model.fit(epochs=epochs)
                            model.save_node_embeddings("{}/output/{}/node_embeddings/node-embeddings_{}_{}_{}_{}.csv".format(dir, exp, method, group, subgroup, iteration))
                        
                        elif method == "dgi-indu":
                            dataset = CustomDataset(nodes_data, edges_data, transform=None)
                            model = DGI_Inductive(dataset, dimension, device)
                            model.fit(epochs=epochs)
                            model.save_node_embeddings("{}/output/{}/node_embeddings/node-embeddings_{}_{}_{}_{}.csv".format(dir, exp, method, group, subgroup, iteration))
                            
                        elif method == "vgae old":
                            data = CustomDatasetVGAE("g_{}_{}".format(group, subgroup), nodes_data, edges_data)
                            graph = data[0]

                            # train
                            args_ = args_vgae(dimension)
                            train_vgae(exp, graph, args_, method, group, subgroup, iteration)
                            
                        elif method == "vgae":
                            transform = T.Compose([
                                # T.NormalizeFeatures(), #
                                T.ToDevice(device),
                                T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True, add_negative_train_samples=False),
                            ])
                            dataset = CustomDataset(nodes_data, edges_data, transform=transform)
                            model = VGAE_Base(dataset, dimension, device)
                            model.fit(epochs=epochs)
                            model.save_node_embeddings("{}/output/{}/node_embeddings/node-embeddings_{}_{}_{}_{}.csv".format(dir, exp, method, group, subgroup, iteration))

                        elif method == "vgae-line":
                            transform = T.Compose([
                                # T.NormalizeFeatures(), #
                                T.ToDevice(device),
                                T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True, add_negative_train_samples=False),
                            ])
                            dataset = CustomDataset(nodes_data, edges_data, transform=transform)
                            model = VGAE_Linear(dataset, dimension, device)
                            model.fit(epochs=epochs)
                            model.save_node_embeddings("{}/output/{}/node_embeddings/node-embeddings_{}_{}_{}_{}.csv".format(dir, exp, method, group, subgroup, iteration))
                        
                        elif method == "argva-base":
                            transform = T.Compose([
                                # T.NormalizeFeatures(), #
                                T.ToDevice(device),
                                T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                                                split_labels=True, add_negative_train_samples=False),
                            ])
                            dataset = CustomDataset(nodes_data, edges_data, transform=transform)
                            model = ARGVA_Base(dataset, dimension, device)
                            model.fit(epochs=epochs)
                            model.save_node_embeddings("{}/output/{}/node_embeddings/node-embeddings_{}_{}_{}_{}.csv".format(dir, exp, method, group, subgroup, iteration))
                    
                # ---
                # Edge embeddings
                # ---
                subgroups_id = subgroups_id_.copy()
                print(method, data_variation)
                
                if data_variation != "none":
                    subgroups_id_op = {}
                    for group in groups_id:
                        subgroups_id_op[group] = [data_variation]
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
                    
                    df_edge_embeddings_concat.to_csv("{}/output/{}/edge_embeddings/edge-embeddings_concat_{}_{}_{}_{}.csv".format(dir, exp, method, group, data_variation, iteration), index=False)
                        
                # outlier detection (ECOD)
                # dict_df_edge_embeddings_concat_outlier = {}
                dict_df_edge_embeddings_concat_filter = {}

                for group in tqdm(groups_id):
                    df_edge_embeddings_concat = pd.read_csv("{}/output/{}/edge_embeddings/edge-embeddings_concat_{}_{}_{}_{}.csv".format(dir, exp, method, group, data_variation, iteration))

                    X_train = df_edge_embeddings_concat.iloc[:, 2:-1]

                    clf = ECOD()
                    clf.fit(X_train)

                    X_train["labels"] = clf.labels_ # binary labels (0: inliers, 1: outliers)

                    df_edge_embeddings_concat_filter = df_edge_embeddings_concat.copy()
                    df_edge_embeddings_concat_filter["labels"] = clf.labels_

                    # save
                    df_edge_embeddings_concat_filter.to_csv("{}/output/{}/edge_embeddings/edge-embeddings_concat_outlier_{}_{}_{}_{}.csv".format(dir, exp, method, group, data_variation, iteration), index=False)
                    
                    df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter[df_edge_embeddings_concat_filter["labels"] == 0].copy()
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
                        df_edge_embeddings_concat_filter_aux = df_edge_embeddings_concat_filter[df_edge_embeddings_concat_filter["subgroup"] == k].copy()
                        
                        # print(df_edge_embeddings_concat_filter)
                        df_edge_embeddings_concat_filter_aux["source"] = df_edge_embeddings_concat_filter_aux["source"].map(dict_id)
                        df_edge_embeddings_concat_filter_aux["target"] = df_edge_embeddings_concat_filter_aux["target"].map(dict_id)
                        df_aux = pd.concat([df_aux, df_edge_embeddings_concat_filter_aux])
                        k += 1
                    dict_df_edge_embeddings_concat_filter[group] = df_aux
                    
                # format id
                if data_variation != "none":
                    for group in tqdm(groups_id):
                        # format
                        df_edge_embeddings_concat_filter = dict_df_edge_embeddings_concat_filter[group]
                        df_edge_embeddings_concat_filter["source"] = df_edge_embeddings_concat_filter["source"].map(lambda x: int(x[1:]))
                        df_edge_embeddings_concat_filter["target"] = df_edge_embeddings_concat_filter["target"].map(lambda x: int(x[1:]))
                            
                # filter by different edges
                if data_variation != "none":
                    for group in tqdm(groups_id):
                        df_edge_embeddings_concat_filter = dict_df_edge_embeddings_concat_filter[group]
                        df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter[df_edge_embeddings_concat_filter["source"] != df_edge_embeddings_concat_filter["target"]].copy()
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
                    df_edge_embeddings_concat_filter.to_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}_{}.csv".format(dir, exp, method, group, data_variation, iteration), index=False)


    # In[21]:


    # import IPython
    # IPython.Application.instance().kernel.do_shutdown(True)


    # In[22]:


    # join
    list_details = []

    for method in methods:
        for k, group in enumerate(groups_id): #
            dict_df_edges_filter = {}
            dict_df_corr = {}
            dict_df_edges_filter_weight = {}
        
            for data_variation in data_variations:
                list_common_subgraph = []
                for iteration in range(iterations):
                    df_edges_filter_weight_filter = pd.read_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}_{}.csv".format(dir, exp, method, group, data_variation, iteration))
                    # print(df_edges_filter_weight_filter)

                    G = nx.from_pandas_edgelist(df_edges_filter_weight_filter) # last change: create_using=nx.Graph() # , edge_attr=["weight"])
                    # SG = G.subgraph([0, 1, 2, 3, 4, 5])
                    # graph_partial_detail(SG, edges=True)
                    list_common_subgraph.append(G)
                    
                print("Union")
                # union
                U = nx.compose_all(list_common_subgraph)
                
                df_compose_subgraph = nx.to_pandas_edgelist(U)
                dict_df_edges_filter[group] = df_compose_subgraph.iloc[:, [0, 1]]
                
                # new correlation
                nodes = list(U.nodes())
                # print(len(nodes)) #
                
                df_join_raw_filter = df_join_raw_log.loc[nodes, :]
                # check_dataset(df_join_raw_filter) #
                # print(df_join_raw_filter.describe()) #
                df_join_raw_filter = df_join_raw_filter.filter(like=group, axis=1)

                df_join_raw_filter_t= df_join_raw_filter.T
                check_dataset(df_join_raw_filter_t)
                # df_join_raw_filter_corr = df_join_raw_filter_t.corr(method="pearson")
                df_join_raw_filter_corr = pg.pcorr(df_join_raw_filter_t)
                check_dataset(df_join_raw_filter_corr)
                dict_df_corr[group] = df_join_raw_filter_corr
                
                # get new correlation
                df_edges_filter_weight = dict_df_edges_filter[group].copy()
                df_corr = dict_df_corr[group]

                df_edges_filter_weight["weight"] = df_edges_filter_weight.apply(lambda x: df_corr.loc[x["source"], x["target"]], axis=1)
                df_edges_filter_weight.sort_values(["source", "target"], ascending=True, inplace=True)
                dict_df_edges_filter_weight[group] = df_edges_filter_weight
                
                # common subgraph
                df_edges_filter_weight = dict_df_edges_filter_weight[group]
                # G = nx.from_pandas_edgelist(df_edges_filter_weight, "source", "target", edge_attr="weight")
                print(method, group, data_variation)
                # print("Before")
                # graph_partial_detail(G, edges=True)
                    
                # filter by abs(weight) >= threshold
                df_edges_filter_weight = dict_df_edges_filter_weight[group]
                df_edges_filter_weight_filter = df_edges_filter_weight[df_edges_filter_weight["weight"].abs() >= threshold_corr]
                df_edges_filter_weight_filter.to_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}.csv".format(dir, exp, method, group, data_variation), index=False)
                
                # print("After")
                # graph_partial_detail(G, edges=True)
                G = nx.from_pandas_edgelist(df_edges_filter_weight_filter, "source", "target", edge_attr="weight")
                list_details.append([method, group, data_variation, G.number_of_nodes(), G.number_of_edges(), nx.density(G)])

    df_details = pd.DataFrame(list_details, columns=["Method", "Group", "Data var.", "Num. nodes", "Num. edges", "Density"])
    df_details.to_csv("{}/output/{}/common_edges/summary.csv".format(dir, exp), index=False)

