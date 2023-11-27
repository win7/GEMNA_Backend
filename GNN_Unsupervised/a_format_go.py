#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[42]:


import json
import os

import pandas as pd

from GNN_Unsupervised.utils.utils_go import *

# %load_ext autotime

dir = os.getcwd() + "/GNN_Unsupervised"
print(dir)

def main(experiment):
    # ### Parameters

    # In[43]:


    params = {
        "methods": [experiment.method], # ["vgae", "dgi", "dgi-tran", "dgi-indu", "argva-base", "vgae-line", "vgae-base"], # ["vgae", "dgi", "dgi-tran", "dgi-indu", "argva-base", "vgae-line", "vgae-base"]
        "data_variations": [experiment.data_variation], # ["none", "str", "dyn"]
        "has_transformation": experiment.transformation,
        "control": experiment.control,
        "dimension": experiment.dimension,
        "threshold_corr": experiment.threshold_corr,
        "threshold_log2": experiment.threshold_log2,
        "alpha": experiment.alpha,
        "iterations": 2,
        "raw_data_file": experiment.raw_data.file.name,
        "obs": "",
        "seeds": [42, 43, 44, 45, 46]
    }
    print(params)


    # In[44]:


    """ dir_path = "output"

    res = sorted(os.listdir(dir_path))
    n = len(res)
    exp = "exp{}".format(n) """
    exp = str(experiment.id)
    exp


    # ### Load dataset

    # In[45]:


    # load dataset groups
    df_raw = pd.read_csv("{}".format(params["raw_data_file"]), delimiter="|")
    # df_raw = pd.read_excel("raw_data/{}.xlsx".format(params["raw_data_file"]))
    df_raw


    # ### Format dataset

    # In[46]:


    # has transformation
    columns_data = list(df_raw.columns)[3:]
    if params["has_transformation"]:
        print("transformation")
        for column in columns_data:
            df_raw[column] = df_raw[column].apply(lambda x: 10**x)
    df_raw


    # In[47]:


    # concat
    df_join_raw = pd.concat([
        df_raw.iloc[:, :]], axis=1)
    df_join_raw.set_index("Alignment ID", inplace=True)
    df_join_raw


    # In[48]:


    # split
    df_join_raw = df_join_raw.rename_axis(None)
    # df_join_raw = df_join_raw.iloc[:, 2:]
    df_join_raw


    # In[49]:


    # get groups name
    groups_id = []
    for item in df_join_raw.iloc[:, 2:].columns.values:
        group_id = item.split("_")[0]
        if group_id not in groups_id:
            groups_id.append(group_id)
    groups_id


    # In[50]:


    # get subgroups names
    subgroups_id = get_subgroups_id(df_join_raw, groups_id)
    subgroups_id


    # In[51]:


    # get groups combination
    groups = []
    control = params["control"]

    groups = []
    properties = list(subgroups_id.keys())
    properties.remove(control)
    for group in properties:
        aux = [control, group]
        groups.append(aux)


    # ### Create folders

    # In[52]:


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

    # In[53]:


    # save dataset
    df_join_raw.to_csv("{}/input/{}_raw.csv".format(dir, exp), index=True)

    # save parameters
    parameters = {
        "exp": exp,
        "methods": params["methods"],
        "data_variations": params["data_variations"],
        "has_transformation": params["has_transformation"],
        "control": params["control"],
        "dimension": params["dimension"],
        "threshold_corr": params["threshold_corr"],
        "threshold_log2": params["threshold_log2"],
        "alpha": params["alpha"],
        "iterations": params["iterations"],
        
        "groups_id": groups_id,
        "subgroups_id": subgroups_id,
        "groups": groups,

        "seeds": params["seeds"],
        "obs": params["obs"]
    }

    with open("{}/output/{}/parameters.json".format(dir, exp), "w") as outfile:
        json.dump(parameters, outfile, indent=4)


    # In[54]:


    """ experiments = {
        "exp": exp
    }

    with open("exp.json".format(dir, experiments), "w") as outfile:
        json.dump(experiments, outfile, indent=4) """

