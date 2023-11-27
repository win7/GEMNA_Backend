#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[36]:


import json

import networkx as nx
import pandas as pd

from GNN_Unsupervised.utils.utils_go import *

# %load_ext autotime

dir = os.getcwd() + "/GNN_Unsupervised"
print(dir)

def main(experiment):
    # file = open("exp.json")
    # experiment = json.load(file)
    # exp_num = experiment["exp"]
    # ### Parameters

    # In[37]:


    """ file = open("exp.json")
    experiment = json.load(file)
    exp_num = experiment["exp"] """
    exp_num = str(experiment.id) # experiment["exp"]

    file = open("{}/output/{}/parameters.json".format(dir, exp_num))
    params = json.load(file)

    exp = params["exp"]
    print("Exp:\t\t", exp)

    data_variations = params["data_variations"]
    print("Data variations:", data_variations)

    threshold_corr = params["threshold_corr"]
    print("Threshold corr:\t", threshold_corr)

    groups_id = params["groups_id"]
    print("Groups id:\t", groups_id)

    subgroups_id = params["subgroups_id"]
    print("Subgroups id:\t", subgroups_id)


    # ### Load dataset

    # In[38]:


    # load dataset groups
    df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(dir, exp), index_col=0)
    df_join_raw = df_join_raw.iloc[:, 2:]
    df_join_raw


    # In[39]:


    check_dataset(df_join_raw)


    # ### Generate graphs

    # In[40]:


    # logarithm
    """ if has_transformation == "true":
        df_join_raw_log = log10_global(df_join_raw)
    else:
        df_join_raw_log = df_join_raw.copy() """

    df_join_raw_log = log10_global(df_join_raw)
    df_join_raw_log.head()


    # In[41]:


    check_dataset(df_join_raw_log)


    # In[42]:


    # split graph in groups and subgroups

    list_df_groups_subgroups = split_groups_subgroups(df_join_raw_log, groups_id, subgroups_id)
    list_df_groups_subgroups[0][0].head()


    # In[43]:


    check_dataset(list_df_groups_subgroups[0][0])


    # In[44]:


    # transpose
    list_groups_subgroups_t = transpose_global(list_df_groups_subgroups)
    list_groups_subgroups_t[0][0]


    # In[45]:


    check_dataset(list_groups_subgroups_t[0][0])


    # In[46]:


    # correlation matrix

    list_groups_subgroups_t_corr = correlation_global(exp, groups_id, subgroups_id, list_groups_subgroups_t, method="pearson", plot=True)
    list_groups_subgroups_t_corr[0][0].head()


    # In[47]:


    check_dataset(list_groups_subgroups_t_corr[0][0])


    # In[48]:


    # build graph (corpus graphs)

    # list_groups_subgroups_t_corr_g = build_graph_weight_global(exp, list_groups_subgroups_t_corr, groups_id, subgroups_id, threshold=0.5)
    list_groups_subgroups_t_corr_g = build_graph_weight_global_directed(exp, list_groups_subgroups_t_corr, groups_id, subgroups_id, threshold=threshold_corr)
    # list_groups_subgroups_t_corr_g = build_graph_weight_global_undirected(exp, list_groups_subgroups_t_corr, groups_id, subgroups_id, threshold=threshold_corr)
    list_groups_subgroups_t_corr_g[0][0]


    # In[49]:


    # create dataset - nodes/edge data for DGL framework

    for data_variation in data_variations:
        if data_variation == "none":
            create_graph_data_go_features_directed(exp, groups_id, subgroups_id, list_df_groups_subgroups)
            # create_graph_data_go_features_undirected(exp, groups_id, subgroups_id, list_df_groups_subgroups)
        else:
            # dynamic graph to static graph
            create_graph_data_other_go_features(exp, groups_id, subgroups_id, data_variation, list_df_groups_subgroups)


    # In[50]:


    # details
    list_details = []
        
    for group in groups_id:
        subgroups = []
        for data_variation in data_variations:
            if data_variation == "none":
                subgroups += subgroups_id[group]
            else:
                subgroups += [data_variation]
        # print(subgroups)
        
        for subgroup in subgroups:
            df_edges = pd.read_csv("{}/output/{}/preprocessing/graphs_data/edges_data_{}_{}.csv".format(dir, exp, group, subgroup))

            G = nx.from_pandas_edgelist(df_edges.iloc[:, [0, 1]])
            list_details.append([group, subgroup, G.number_of_nodes(), G.number_of_edges(), nx.density(G)])

    df_details = pd.DataFrame(list_details, columns=["Group", "Subgroup", "Num. nodes", "Num. edges", "Density"])
    df_details.to_csv("{}/output/{}/preprocessing/graphs_data/summary.csv".format(dir, exp), index=False)

    """ df_details = pd.read_csv("output/{}/preprocessing/graphs_data/summary.csv".format(exp))
    df_details """

