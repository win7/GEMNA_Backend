# ### Imports
import os
import sys

dir = os.getcwd() + "/GNN_Unsupervised"
sys.path.append(dir)

from tqdm import tqdm
from utils.utils import *

import networkx as nx
import numpy as np
import pandas as pd
import os
import sys

# %load_ext autotime

import json

def main(experiment):
    # ### Parameters
    # Opening JSON file
    # print(dir)
    file = open("{}/input/parameters_{}.json".format(dir, experiment.id))
    params = json.load(file)

    # dir = os.path.dirname(os.getcwd())
    # print(dir)

    exp = params["exp"]
    print("Exp:\t\t", exp)

    groups_id = params["groups_id"]
    print("Groups id:\t", groups_id)

    subgroups_id = params["subgroups_id"]
    print("Subgroups id:\t", subgroups_id)

    option = params["option"]
    print("Option:\t", option)

    # ### Load dataset
    # load dataset groups
    df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(dir, exp), index_col=0)
    df_join_raw = df_join_raw.iloc[:, 2:]

    # ### Generate graphs
    # logarithm
    if experiment.transformation == "true":
        df_join_raw_log = log10_global(df_join_raw)
    else:
        df_join_raw_log = df_join_raw.copy()

    # split graph in groups and subgroups
    list_df_groups_subgroups = split_groups_subgroups(df_join_raw_log, groups_id, subgroups_id)

    # transpose
    list_groups_subgroups_t = transpose_global(list_df_groups_subgroups)

    # correlation matrix
    list_groups_subgroups_t_corr = correlation_global(exp, groups_id, subgroups_id, list_groups_subgroups_t, method="pearson", plot=True)

    # build graph
    # list_groups_subgroups_t_corr_g = build_graph_weight_global(exp, list_groups_subgroups_t_corr, groups_id, subgroups_id, threshold=0.5)
    list_groups_subgroups_t_corr_g = build_graph_weight_global_(exp, list_groups_subgroups_t_corr, groups_id, subgroups_id, threshold=0.5)

    # create dataset - nodes/edge data for DGL framework
    if option == "none":
        create_graph_data(exp, groups_id, subgroups_id)
    else:
        create_graph_data_other(exp, groups_id, subgroups_id, option=option)