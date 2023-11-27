#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[62]:


from tqdm import tqdm
from utils.utils_go import *

import networkx as nx
import numpy as np
import pandas as pd
import os
import sys
import json

# get_ipython().run_line_magic('load_ext', 'autotime')

dir = os.getcwd() + "/GNN_Unsupervised"
print(dir)

# ### Parameters

# In[63]:


def main(experiment):
    # file = open("exp.json")
    # experiment = json.load(file)
    exp_num = str(experiment.id) # experiment["exp"]

    file = open("{}/output/{}/parameters.json".format(dir, exp_num))
    params = json.load(file)

    exp = params["exp"]
    print("Exp:\t\t", exp)

    groups_id = params["groups_id"]
    print("Groups id:\t", groups_id)

    subgroups_id = params["subgroups_id"]
    print("Subgroups id:\t", subgroups_id)

    option = params["option"]
    print("Option:\t\t", option)

    """ has_transformation = params["transformation"]
    print("Transformation:\t", has_transformation) """

    threshold = params["threshold"]
    print("Threshold:\t", threshold)


    # ### Load dataset

    # In[64]:


    # load dataset groups
    df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(dir, exp), index_col=0)
    df_join_raw = df_join_raw.iloc[:, 2:]
    df_join_raw


    # In[65]:


    df_join_raw.columns


    # ### Generate graphs

    # In[66]:


    # logarithm
    """ if has_transformation == "true":
        df_join_raw_log = log10_global(df_join_raw)
    else:
        df_join_raw_log = df_join_raw.copy() """

    df_join_raw_log = log10_global(df_join_raw)
    df_join_raw_log.head()


    # In[67]:


    # split graph in groups and subgroups

    list_df_groups_subgroups = split_groups_subgroups(df_join_raw_log, groups_id, subgroups_id)
    list_df_groups_subgroups[0][1].head()


    # In[68]:


    # temp
    list_df_groups_subgroups[0][1]


    # In[69]:


    # transpose
    list_groups_subgroups_t = transpose_global(list_df_groups_subgroups)
    list_groups_subgroups_t[0][0]


    # In[70]:


    # correlation matrix

    list_groups_subgroups_t_corr = correlation_global(exp, groups_id, subgroups_id, list_groups_subgroups_t, method="pearson", plot=True)
    list_groups_subgroups_t_corr[0][0].head()


    # In[71]:


    # build graph (corpus graphs)
    # threshold = 0.01
    # list_groups_subgroups_t_corr_g = build_graph_weight_global(exp, list_groups_subgroups_t_corr, groups_id, subgroups_id, threshold=0.5)
    list_groups_subgroups_t_corr_g = build_graph_weight_global_(exp, list_groups_subgroups_t_corr, groups_id, subgroups_id, threshold=threshold)
    list_groups_subgroups_t_corr_g[0][0]


    # In[72]:


    # partial correlation

    """ list_groups_subgroups_t_corr_g = build_graph_weight_global_partial_corr(exp, groups_id, subgroups_id, list_groups_subgroups_t, threshold=threshold)
    list_groups_subgroups_t_corr_g[0][0] """


    # In[73]:


    # create dataset - nodes/edge data for DGL framework

    create_graph_data_go_features(exp, groups_id, subgroups_id, list_df_groups_subgroups)


    # ### Dynamic graph to Static graph

    # In[74]:


    create_graph_data_other_go_features(exp, groups_id, subgroups_id, "str", list_df_groups_subgroups)


    # In[75]:


    create_graph_data_other_go_features(exp, groups_id, subgroups_id, "dyn", list_df_groups_subgroups)


    # In[76]:

