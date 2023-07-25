# %% [markdown]
# ### Imports

# %%
import os
import sys

dir = os.getcwd() + "/GNN_Unsupervised"
sys.path.append(dir)

from tqdm import tqdm
from utils.utils import *

import json
import networkx as nx
import numpy as np
import pandas as pd
import os
import sys

# %load_ext autotime

# %% [markdown]
# ### Parameters

# %%

def main(exp, raw_data_file, method, option, dimension):
    """ dir_path = "output"

    res = sorted(os.listdir(dir_path))
    n = len(res) + 1
    exp = "exp{}".format(n)
    exp

    # %%
    method = "dgi" # vgae, dgi
    dimension = 3
    option = "dyn" # dyn, str
    raw_data_folder = "Edwin_proyecto3" # change
    raw_data_file = "Trial 1_Reinhard" # change """

    # %% [markdown]
    # ### Load dataset

    # %%
    # load dataset groups
    df1 = pd.read_csv("{}".format(raw_data_file), delimiter="|")
    df1

    # %%
    # drop duplicates
    df1.drop_duplicates(subset=["Id"], keep="last", inplace=True)
    df1

    # %% [markdown]
    # ### Format dataset

    # %%
    # concat
    df_join_raw = pd.concat([
        df1.iloc[:, :]], axis=1)
    df_join_raw.set_index("Id", inplace=True)

    # print(df_join_raw.shape)
    df_join_raw = df_join_raw.rename_axis(None)
    df_join_raw = df_join_raw.iloc[:, 0:]
    df_join_raw

    # %%
    # get groups name
    groups_id = []
    for item in df_join_raw.iloc[:, 1:].columns.values:
        group_id = item.split("_")[0]
        if group_id not in groups_id:
            groups_id.append(group_id)
    groups_id

    # %%
    # change columns name (AB1.1)

    """ columns = df_join_raw.columns.values

    for i in range(len(columns)):
        # remove white space
        columns[i] = columns[i].replace(" ", "")
        columns[i] = columns[i].split("/")[1] 

    df_join_raw.columns = columns
    df_join_raw """

    # %%
    # get subgroups names
    subgroups_id = get_subgroups_id(df_join_raw, groups_id)
    subgroups_id

    # %%
    # get options
    options = {}

    for group in groups_id:
        options[group] = [option]

    # %% [markdown]
    # ### Save dataset and parameters

    # %%
    # save dataset
    df_join_raw.to_csv("{}/input/{}_raw.csv".format(dir, exp), index=True)

    # save parameters

    # Data to be written
    parameters = {
        # "raw_folder": raw_data_folder,
        "exp": exp,
        "method": method,
        "dimension": dimension,
        "groups_id": groups_id,
        "subgroups_id": subgroups_id,
        "option": option
    }
    
    json_object = json.dumps(parameters)
    with open("{}/input/parameters_{}.json".format(dir, exp), "w") as outfile:
        json.dump(parameters, outfile, indent=4)

    # %% [markdown]
    # ### Create folders

    # %%
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