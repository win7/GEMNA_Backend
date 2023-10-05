#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[4]:


from mpl_toolkits import mplot3d
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
from utils.utils_go import *

import json
import hdbscan
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import sys

# get_ipython().run_line_magic('load_ext', 'autotime')


# ### Parameters

# In[5]:


dir = os.getcwd() + "/GNN_Unsupervised"
print(dir)

def main(experiment):
    # file = open("exp.json")
    # experiment = json.load(file)
    exp_num = str(experiment.id) # experiment["exp"]

    file = open("{}/output/exp{}/parameters.json".format(dir, exp_num))
    params = json.load(file)

    exp = params["exp"]
    print("Exp:\t\t", exp)

    method = params["method"]
    print("Method:\t\t", method)

    methods = params["methods"]
    print("Methods:\t", methods)

    groups_id = params["groups_id"]
    groups = [groups_id[2], groups_id[0]]
    print("Groups:\t\t", groups)

    subgroups_id = params["subgroups_id"]
    print("Subgroups id:\t", subgroups_id)

    option = params["option"]
    print("Option:\t\t", option)

    options = params["options"]
    print("Options:\t", options)

    control = params["control"]
    print("Control:\t", control)

    list_groups = []
    propy = list(subgroups_id.keys())
    propy.remove(control)
    for group in propy:
        aux = [control, group]
        list_groups.append(aux)
        
    # print(list_groups)
    # groups = list_groups[1] # change
    print("Groups:\t\t", list_groups)


    # ### Biocyc

    # In[6]:


    # load raw data
    df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(dir, exp), index_col=0)

    for method in methods[:1]: # change
        for option in options[:1]: # change
                print(method, option)
                for groups in list_groups: # change
                    # get common nodes from change detection result
                    df_change_filter = pd.read_csv("{}/output/{}/changes/changes_edges_log2_{}_{}_{}_{}.csv".format(dir, exp, method, groups[0], groups[1], option))
                    
                    G = nx.from_pandas_edgelist(df_change_filter.iloc[:, [0, 1]])
                    nodes = list(G.nodes())
                    
                    # mapping metabolite name with ratio (2)
                    df_biocyc = pd.DataFrame()
                    df_biocyc["Alignment ID"] = nodes
                    list_data = []
                    for group in groups:
                        df_aux = df_join_raw.filter(like=group)
                        df_aux = df_aux.loc[nodes]
                        # df_biocyc["mean-{}".format(group)] = df_aux.mean(axis=1).values
                        # df_biocyc["log-{}".format(group)] = np.log10(df_aux.mean(axis=1).values)
                        list_data.append(df_aux.mean(axis=1).values)
                    df_biocyc[groups[0]] = np.log10(list_data[0])
                    df_biocyc[groups[1]] = np.log10(list_data[1])
                    df_biocyc["Ratio"] = np.log2(np.divide(list_data[1], list_data[0]))
                    # df_biocyc["metabolities"] = df_metadata.loc[common_nodes]["Metabolites - Approved by Nicola"].values
                    df_biocyc.insert(0, "Average Mz", df_join_raw.loc[nodes]["Average Mz"].values)
                    df_biocyc.insert(0, "Metabolite name", df_join_raw.loc[nodes]["Metabolite name"].values)         
                    # df_biocyc = df_biocyc.iloc[:, 1:]
                    df_biocyc.to_csv("{}/output/{}/biocyc/biocyc_{}_{}_{}.csv".format(dir, exp, method, "-".join(groups), option), index=False, sep="\t") # header=False
                    
                    df_biocyc = pd.read_csv("{}/output/{}/biocyc/biocyc_{}_{}_{}.csv".format(dir, exp, method, "-".join(groups), option),
                            names=["Metabolite name", "Average Mz", "Alignment ID", groups[0], groups[1], "Ratio"], sep="\t") # header=None
                    
                    # plot
                    fig = go.Figure(data=go.Heatmap(
                    z=df_biocyc.iloc[:, 3:5].T.values,
                    y=groups,
                    x=list(map(str, df_biocyc.iloc[:, 1].values)),
                    hoverongaps = False))
                    fig.show()
                    
                    fig = go.Figure(data=go.Heatmap(
                    z=df_biocyc.iloc[:, -1:].T.values,
                    y=["ratio"],
                    x=list(map(str, df_biocyc.iloc[:, 1].values)),
                    hoverongaps = False))
                    fig.show()

