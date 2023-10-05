#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[4]:


from utils.utils_go_ import *

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import scipy
import sys

# get_ipython().run_line_magic('load_ext', 'autotime')


# ### Parameters

# In[5]:


import json
  
file = open("exp.json")
experiment = json.load(file)
exp_num = experiment["exp"]

file = open("output/exp{}/parameters.json".format(exp_num))
params = json.load(file)

exp = params["exp"]
print("Exp:\t\t", exp)

method = params["method"]
print("Method:\t\t", method)

methods = params["methods"]
print("Methods:\t", methods)

alpha = params["alpha"]
print("Alpha:\t\t", alpha)

groups_id = params["groups_id"]
print("Groups id:\t", groups_id)

subgroups_id = params["subgroups_id"]
print("Subgroups id:\t", subgroups_id)

groups = [groups_id[2], groups_id[0]]
print("Groups:\t\t", groups)

subgroups = {groups[0]: subgroups_id[groups[0]], groups[1]: subgroups_id[groups[1]]}
print("Subgroups id:\t", subgroups)

option = params["option"]
print("Option:\t\t", option)

options = params["options"]
print("Options:\t", options)

threshold_log2 = params["threshold_log2"]
print("Threshold log2:\t", threshold_log2)

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


# ### Changes detection

# In[6]:


# read raw data
df_join_raw = pd.read_csv("input/{}_raw.csv".format(exp), index_col=0)
df_join_raw.head()

for method in methods[:1]: # change
    for option in options[:1]: # change
            print(method, option)
            for groups in list_groups: # change
                # read edges
                list_graphs = []
                for k in range(len(groups)):
                    df_edges = pd.read_csv("output/{}/common_edges/common_edges_{}_{}_{}.csv".format(exp, method, groups[k], option))
                    sort_df_edges(df_edges)
                    df_edges.rename(columns={"weight": "weight{}".format(k + 1)}, inplace=True)
                    G = nx.from_pandas_edgelist(df_edges, edge_attr=["weight{}".format(k + 1)])
                    list_graphs.append(G)
                    
                # compose
                R = nx.compose(list_graphs[0], list_graphs[1])
                nx.set_edge_attributes(R, {(u, v): {"label": get_label(ed, th=0.8)} for u, v, ed in R.edges.data()})
                # print(R.number_of_nodes(), R.number_of_edges())

                df_change = nx.to_pandas_edgelist(R)
                df_change = df_change[["source", "target", "weight1", "weight2", "label"]]
                df_change.to_csv("output/{}/changes/changes_edges_compose_{}_{}_{}_{}.csv".format(exp, method, groups[0], groups[1], option), index=False)

                # differences between correlations
                df_change.insert(3, "N1", [len(df_join_raw.filter(like=groups[0]).columns)] * len(df_change))
                df_change.insert(5, "N2", [len(df_join_raw.filter(like=groups[1]).columns)] * len(df_change))

                # differences between correlations
                n1 = len(df_join_raw.filter(like=groups[0]).columns) # len(df_change) - df_change["weight1"].isna().sum() # len(df_change)
                n2 = len(df_join_raw.filter(like=groups[1]).columns) # len(df_change) - df_change["weight2"].isna().sum() # len(df_change)
                z1 = fisher_transform(df_change["weight1"])
                z2 = fisher_transform(df_change["weight2"])
                sezdiff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
                diff = z1 - z2
                ztest = diff / sezdiff
                p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(ztest), loc=0, scale=1))
                df_change["p-value"] = p_value

                # get sigtnificant
                list_significant = []
                for row in df_change.itertuples():
                    # print(row[3], row[5], row[8])
                    if row[8] < alpha:
                        list_significant.append("*")
                    elif row[8] >= alpha:
                        list_significant.append("")
                    elif np.isnan(row[3]) or np.isnan(row[5]):
                        list_significant.append("*")
                    else:
                        list_significant.append("x")
                df_change["significant"] = list_significant
                # count_values(df_change["significant"])

                try:
                    x = df_change["p-value"]
                    hist(x, th=alpha)
                except:
                    pass

                # filter by significant
                # df_change_filter = df_change[df_change["p-value"] < alpha]
                df_change_filter1 = df_change[df_change["significant"] != "x"]
                df_change_filter1.to_csv("output/{}/changes/changes_edges_significant_{}_{}_{}_{}.csv".format(exp, method, groups[0], groups[1], option), index=False)

                # common subgraph
                G = nx.from_pandas_edgelist(df_change_filter1.iloc[:, [0, 1]])

                # filter raw data by common nodes
                nodes_common = sorted(list(G.nodes()))
                df_join_raw_filter = df_join_raw.loc[nodes_common]

                # filter by log2
                df_join_raw_filter_log2 = np.log2(df_join_raw_filter.filter(like=groups[1]).mean(axis=1) / df_join_raw_filter.filter(like=groups[0]).mean(axis=1))
                df_join_raw_filter_log2 = df_join_raw_filter_log2.to_frame()
                df_join_raw_filter_log2.columns = ["log2"]

                df_join_raw_filter_log2_filter = df_join_raw_filter_log2[((df_join_raw_filter_log2["log2"] > threshold_log2) | (df_join_raw_filter_log2["log2"] < -threshold_log2))]
                nodes_log2 = list(df_join_raw_filter_log2_filter.index)

                log2 = []
                for row in df_change_filter1.itertuples():
                    if row[1] in nodes_log2 and row[2] in nodes_log2:
                        log2.append(True)
                    else:
                        log2.append(False)
                df_change_filter1["log2"] = log2

                df_change_filter2 = df_change_filter1[df_change_filter1["log2"] == True]
                df_change_filter2 = df_change_filter2.iloc[:, :-1]

                # mapping aligment ID to average mz
                df_change_filter = df_change_filter2.copy() # df_change_filter1, df_change_filter2

                dict_aux = df_join_raw.iloc[:, :2].to_dict(orient="dict")
                dict_mz = dict_aux["Average Mz"]
                dict_mz = {key: value for key, value in dict_mz.items()}
                dict_metabolite = dict_aux["Metabolite name"]
                dict_metabolite = {key: value for key, value in dict_metabolite.items()}

                # mapping
                df_change_filter.insert(len(df_change_filter.columns), "source1", df_change_filter["source"].map(dict_mz))
                df_change_filter.insert(len(df_change_filter.columns), "target1", df_change_filter["target"].map(dict_mz))
                df_change_filter.insert(len(df_change_filter.columns), "source2", df_change_filter["source"].map(dict_metabolite))
                df_change_filter.insert(len(df_change_filter.columns), "target2", df_change_filter["target"].map(dict_metabolite))
                df_change_filter.to_csv("output/{}/changes/changes_edges_log2_{}_{}_{}_{}.csv".format(exp, method, groups[0], groups[1], option), index=False)

                # get common nodes from change detection result
                df_change_filter = pd.read_csv("output/{}/changes/changes_edges_log2_{}_{}_{}_{}.csv".format(exp, method, groups[0], groups[1], option))
                G = nx.from_pandas_edgelist(df_change_filter, edge_attr=["label"], create_using=nx.DiGraph())
                SG = G.subgraph([0, 1, 2, 3, 4, 5])
                SG = nx.relabel_nodes(SG, id_mz)
                edge_labels = nx.get_edge_attributes(SG, "label")
                try:
                    pos = nx.planar_layout(SG)
                except:
                    # pos = nx.spring_layout(SG, k=0.5*1/np.sqrt(len(SG.nodes())), iterations=10)
                    pos = nx.circular_layout(SG)
                finally:
                    pos = pos=nx.spring_layout(SG) 
                nx.draw_networkx(SG, pos, font_color="black", font_size=10, node_color="orange")
                nx.draw_networkx_edge_labels(SG, pos, edge_labels, font_size=10)
                plt.suptitle("{} --> {}".format(groups[0], groups[1]))
                plt.title("Nodes: {}; Edges: {}".format(SG.number_of_nodes(), SG.number_of_edges()))
                plt.show()

