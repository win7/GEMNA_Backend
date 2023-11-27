# ### Imports
import os
import sys

dir = os.getcwd() + "/GNN_Unsupervised"
sys.path.append(dir)

from tqdm import tqdm
from utils.utils_v1 import *

import json
import networkx as nx
import numpy as np
import pandas as pd
import os
import sys

# %load_ext autotime

# ### Parameters
def main(experiment):
    """ dir_path = "output"

    res = sorted(os.listdir(dir_path))
    n = len(res) + 1
    exp = "exp{}".format(n)
    exp

    method = "dgi" # vgae, dgi
    dimension = 3
    option = "dyn" # dyn, str
    raw_data_folder = "Edwin_proyecto3" # change
    raw_data_file = "Trial 1_Reinhard" # change """

    # load dataset groups
    df_raw = pd.read_csv("{}".format(experiment.raw_data), delimiter="|")
    
    # drop duplicates
    # df1.drop_duplicates(subset=["Id"], keep="last", inplace=True)

    # ### Format dataset
    # has transformation
    columns_data = list(df_raw.columns)[3:]
    if experiment.transformation == "true":
        for column in columns_data:
            df_raw[column] = df_raw[column].apply(lambda x: 10**x)

    # concat
    df_join_raw = pd.concat([
        df_raw.iloc[:, :]], axis=1)
    df_join_raw.set_index("Alignment ID", inplace=True)

    # split
    # print(df_join_raw.shape)
    df_join_raw = df_join_raw.rename_axis(None)
    # df_join_raw = df_join_raw.iloc[:, 2:]

    # get groups name
    groups_id = []
    for item in df_join_raw.iloc[:, 2:].columns.values:
        group_id = item.split("_")[0]
        if group_id not in groups_id:
            groups_id.append(group_id)

    # get subgroups names
    subgroups_id = get_subgroups_id(df_join_raw, groups_id)

    # get options
    options = {}
    for group in groups_id:
        options[group] = [experiment.data_variation]

    # ### Save dataset and parameters
    # save dataset
    df_join_raw.to_csv("{}/input/{}_raw.csv".format(dir, experiment.id), index=True)
    
    # save parameters
    parameters = {
        # "raw_folder": raw_data_folder,
        "exp": str(experiment.id),
        "method": experiment.method,
        "dimension": experiment.dimension,
        "groups_id": groups_id,
        "subgroups_id": subgroups_id,
        "option": experiment.data_variation,
        "control": experiment.control,
        "range": experiment.range,
        "transformation": experiment.transformation,
        "alpha": experiment.alpha,
        "threshold": experiment.threshold_corr,
        "threshold_log2": experiment.threshold_log2
    }

    with open("{}/input/parameters_{}.json".format(dir, experiment.id), "w") as outfile:
        json.dump(parameters, outfile, indent=4)

    # ### Create folders
    # create experiments folder
    try: 
        os.mkdir("{}/output/{}".format(dir, experiment.id))
        os.mkdir("{}/output/{}/correlations".format(dir, experiment.id))
        os.mkdir("{}/output/{}/preprocessing".format(dir, experiment.id))
        os.mkdir("{}/output/{}/preprocessing/edges".format(dir, experiment.id))
        os.mkdir("{}/output/{}/preprocessing/graphs".format(dir, experiment.id))
        os.mkdir("{}/output/{}/preprocessing/graphs_data".format(dir, experiment.id))
        os.mkdir("{}/output/{}/node_embeddings".format(dir, experiment.id))
        os.mkdir("{}/output/{}/edge_embeddings".format(dir, experiment.id))
        os.mkdir("{}/output/{}/common_edges".format(dir, experiment.id))
        os.mkdir("{}/output/{}/changes".format(dir, experiment.id))
        os.mkdir("{}/output/{}/plots".format(dir, experiment.id))
        os.mkdir("{}/output/{}/biocyc".format(dir, experiment.id))
    except OSError as error: 
        print(error)