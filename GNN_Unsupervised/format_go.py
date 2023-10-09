#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[23]:

from tqdm import tqdm
from utils.utils_go import *

import json
import networkx as nx
import numpy as np
import pandas as pd
import os
import sys

# get_ipython().run_line_magic('load_ext', 'autotime')

dir = os.getcwd() + "/GNN_Unsupervised"
print(dir)

def main(experiment):
    # ### Parameters

    # In[24]:

    """ dir_path = "output"

    res = sorted(os.listdir(dir_path))
    n = len(res) + 1
    exp = "exp{}".format(n) """
    exp = str(experiment.id)


    # In[25]:


    method = experiment.method # vgae, dgi
    methods = [experiment.method]#, "vgae"]
    dimension = experiment.dimension
    option = experiment.data_variation # none, dyn, str
    options = ["none", "str", "dyn"]
    control = experiment.control # "T" # "LG" # "FCSglc" # "WT"
    range_ = 10
    has_transformation = experiment.transformation # true or false # change
    alpha = experiment.alpha
    threshold = experiment.threshold_corr
    threshold_log2 = experiment.threshold_log2
    seeds = [42, 43, 44, 45, 46]
    iterations = 2
    raw_data_file = experiment.raw_data # "mutant" # "reinhard" # "mutant_a" # "plant" # "single_cell" # ""Trial Alfredo process" # "Trial 1_Reinhard" # "Trial Alfredo" # change
    obs = ""

    # ### Load dataset

    # In[26]:


    # Alfredo raw data
    """ df_temp = pd.read_excel("/home/ealvarez/Project/GNN_Filter/GNN_unsupervised/raw_data/Edwin_proyecto2/Edwin_Set2-processed.xlsx")
    df_temp """


    # In[27]:


    """ df_temp = df_temp.set_index("ionMz")
    df_temp = df_temp.iloc[:, 4:]
    df_temp """


    # In[28]:


    """ df_temp = df_temp.T
    df_temp """


    # In[29]:


    """ nodes = list(df_temp.index)
    print(nodes) """


    # In[30]:


    """ df_raw_filter = df_raw[df_raw["Average Mz"].isin(nodes)]
    df_raw_filter """


    # In[31]:


    # df_raw_filter.to_csv("Trial Alfredo process.csv", sep="|", index=False)


    # ---

    # In[32]:

    # load dataset groups
    df_raw = pd.read_csv("{}/raw_data/{}.csv".format(dir, raw_data_file), delimiter="|")
    df_raw


    # In[33]:


    # drop duplicates
    # df_raw.drop_duplicates(subset=["Id"], keep="last", inplace=True)
    # df_raw


    # ### Format dataset

    # In[34]:


    # has transformation
    columns_data = list(df_raw.columns)[3:]
    if has_transformation == "true":
        print("transformation")
        for column in columns_data:
            df_raw[column] = df_raw[column].apply(lambda x: 10**x)
    df_raw


    # In[35]:


    # concat
    df_join_raw = pd.concat([
        df_raw.iloc[:, :]], axis=1)
    df_join_raw.set_index("Alignment ID", inplace=True)
    df_join_raw


    # In[36]:


    # split
    df_join_raw = df_join_raw.rename_axis(None)
    # df_join_raw = df_join_raw.iloc[:, 2:]
    df_join_raw


    # In[38]:


    # get groups name
    groups_id = []
    for item in df_join_raw.iloc[:, 2:].columns.values:
        group_id = item.split("_")[0]
        if group_id not in groups_id:
            groups_id.append(group_id)
    groups_id


    # In[39]:


    # change columns name (AB1.1)

    """ columns = df_join_raw.columns.values

    for i in range(len(columns)):
        # remove white space
        columns[i] = columns[i].replace(" ", "")
        columns[i] = columns[i].split("/")[1] 

    df_join_raw.columns = columns
    df_join_raw """


    # In[40]:


    # get subgroups names
    subgroups_id = get_subgroups_id(df_join_raw, groups_id)
    subgroups_id


    # In[41]:


    # get options
    """ options = {}
    for group in groups_id:
        options[group] = [option]
    options """


    # ### Create folders

    # In[42]:


    # create experiments folder
    try: 
        os.mkdir("{}/output/{}".format(dir, exp))
        os.mkdir("{}/output/{}/correlations".format(dir, exp))
        os.mkdir("{}/output/{}/preprocessing".format(dir, exp))
        os.mkdir("{}/output/{}/preprocessing/edges".format(dir, exp))
        os.mkdir("{}/output/{}/preprocessing/graphs".format(dir, exp))
        os.mkdir("{}/output/{}/preprocessing/graphs_data".format(dir, exp))
        os.mkdir("{}/output/{}/node_embeddings".format(dir, exp))
        os.mkdir("{}/output/{}/edge_embeddings".format(dir, exp))
        os.mkdir("{}/output/{}/common_edges".format(dir, exp))
        os.mkdir("{}/output/{}/changes".format(dir, exp))
        os.mkdir("{}/output/{}/plots".format(dir, exp))
        os.mkdir("{}/output/{}/biocyc".format(dir, exp))
    except OSError as error: 
        print(error)


    # ### Save dataset and parameters

    # In[43]:


    # save dataset
    df_join_raw.to_csv("{}/input/{}_raw.csv".format(dir, exp), index=True)

    # save parameters
    parameters = {
        # "raw_folder": raw_data_folder,
        "exp": exp,
        "method": method,
        "methods": methods,
        "dimension": dimension,
        "groups_id": groups_id,
        "subgroups_id": subgroups_id,
        "option": option,
        "options": options,
        "control": control,
        "range": range_,
        "transformation": has_transformation,
        "alpha": alpha,
        "threshold": threshold,
        "threshold_log2": threshold_log2,
        "seeds": seeds,
        "iterations": iterations,
        "obs": obs
    }

    with open("{}/output/{}/parameters.json".format(dir, exp), "w") as outfile:
        json.dump(parameters, outfile, indent=4)

