#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import json

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from GNN_Unsupervised.utils.utils_go import *

# %load_ext autotime

dir = os.getcwd() + "/GNN_Unsupervised"
print(dir)

def main(experiment):
    # ### Parameters

    # In[2]:


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

    control = params["control"]
    print("Control:\t", control)

    subgroups_id = params["subgroups_id"]
    print("Subgroups id:\t", subgroups_id)

    groups = params["groups"]
    print("Groups:\t\t", groups)


    # ### Biocyc

    # In[3]:


    # load raw data
    df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(dir, exp), index_col=0)

    for method in methods: # change
        for option in data_variations: # change
            for group in groups: # change
                # get common nodes from change detection result
                df_change_filter = pd.read_csv("{}/output/{}/changes/changes_edges_log2_{}_{}_{}_{}.csv".format(dir, exp, method, group[0], group[1], option))
                
                G = nx.from_pandas_edgelist(df_change_filter.iloc[:, [0, 1]])
                nodes = list(G.nodes())
                
                # mapping metabolite name with ratio (2)
                df_biocyc = pd.DataFrame()
                df_biocyc["Alignment ID"] = nodes
                list_data = []
                for group_id in group:
                    df_aux = df_join_raw.filter(like=group_id)
                    df_aux = df_aux.loc[nodes]
                    # df_biocyc["mean-{}".format(group)] = df_aux.mean(axis=1).values
                    # df_biocyc["log-{}".format(group)] = np.log10(df_aux.mean(axis=1).values)
                    list_data.append(df_aux.mean(axis=1).values)
                df_biocyc[group[0]] = np.log10(list_data[0])
                df_biocyc[group[1]] = np.log10(list_data[1])
                df_biocyc["Ratio"] = np.log2(np.divide(list_data[1], list_data[0]))
                # df_biocyc["metabolities"] = df_metadata.loc[common_nodes]["Metabolites - Approved by Nicola"].values
                df_biocyc.insert(0, "Average Mz", df_join_raw.loc[nodes]["Average Mz"].values)
                df_biocyc.insert(0, "Metabolite name", df_join_raw.loc[nodes]["Metabolite name"].values)         
                # df_biocyc = df_biocyc.iloc[:, 1:]
                df_biocyc.to_csv("{}/output/{}/biocyc/biocyc_{}_{}_{}.csv".format(dir, exp, method, "-".join(group), option), index=False, sep="\t") # header=False
                
                """ df_biocyc = pd.read_csv("{}/output/{}/biocyc/biocyc_{}_{}_{}.csv".format(dir, exp, method, "-".join(group), option),
                        names=["Metabolite name", "Average Mz", "Alignment ID", group[0], group[1], "Ratio"], sep="\t") # header=None """

