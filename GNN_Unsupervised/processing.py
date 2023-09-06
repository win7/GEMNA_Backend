# ### Imports
import os
import sys

dir = os.getcwd() + "/GNN_Unsupervised"
sys.path.append(dir)

from tqdm import tqdm
from utils.utils import *

from pyod.models.ecod import ECOD
from mpl_toolkits import mplot3d
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
# from utils.utils import *

# import hdbscan
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import sys

import json

def main(experiment):
    # Opening JSON file
    file = open("{}/input/parameters_{}.json".format(dir, experiment.id))
    params = json.load(file)

    exp = params["exp"]
    print("Exp:\t\t", exp)

    method = params["method"]
    print("Method:\t\t", method)

    dimension = params["dimension"]
    print("Dimension:\t", dimension)

    groups_id = params["groups_id"]
    print("Groups id:\t", groups_id)

    subgroups_id = params["subgroups_id"]
    print("Subgroups id:\t", subgroups_id)

    option = params["option"]
    print("Option:\t\t", option)

    threshold = params["threshold"]
    print("Threshold:\t", threshold)

    if option:
        subgroups_id_op = {}
        for group in groups_id:
            subgroups_id_op[group] = [option]
    else:
        subgroups_id_op = subgroups_id
    print("Subgroups id op:", subgroups_id_op)

    # ### Edge embeddings
    # get edges embeddings

    edge_embeddings_global(exp, method, groups_id, subgroups_id_op)

    # ### Concat edge embeddings
    for group in tqdm(groups_id):
        df_edge_embeddings_concat = pd.DataFrame()
        k = 0
        for subgroup in tqdm(subgroups_id_op[group]):
            df_edge_embeddings = pd.read_csv("{}/output/{}/edge_embeddings/edge-embeddings_{}_{}_{}.csv".format(dir, exp, method, group, subgroup))
            df_edge_embeddings["subgroup"] = [k] * len(df_edge_embeddings)
            df_edge_embeddings_concat = pd.concat([df_edge_embeddings_concat, df_edge_embeddings])
            k += 1
        
        df_edge_embeddings_concat.to_csv("{}/output/{}/edge_embeddings/edge-embeddings_concat_{}_{}_{}.csv".format(dir, exp, method, group, option), index=False)

    # df_edge_embeddings_concat = pd.read_csv("{}/output/{}/edge_embeddings/edge-embeddings_concat_{}_{}_{}.csv".format(dir, exp, method, groups_id[0], option))

    # plot edge embeddings concat
    """ for group in tqdm(groups_id):
        df_edge_embeddings_concat = pd.read_csv("{}/output/{}/edge_embeddings/edge-embeddings_concat_{}_{}_{}.csv".format(dir, exp, method, group, option), index_col=[0, 1])

        x = df_edge_embeddings_concat.iloc[:, 0]
        y = df_edge_embeddings_concat.iloc[:, 1]
        z = df_edge_embeddings_concat.iloc[:, 2]

        # Creating figure
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")

        # Creating plot
        ax.scatter3D(x, y, z, c=df_edge_embeddings_concat.iloc[:, -1], alpha=0.1)
        # plt.title("Dimension: {}".format(dimension))

        # show plot
        plt.savefig("{}/output/{}/plots/edge-embeddings_concat_{}_{}_{}.png".format(dir, exp, method, group, option))
        # plt.show()
        plt.clf() """

    # ### Outliers detection
    # Outlier detection (HDBSCAN)

    """ df_edge_embeddings_concat = pd.read_csv("{}/output/edge_embeddings/edge-embeddings_concat_{}_{}_{}_{}.csv".format(group, method, dimension, "L2"), index_col=[0, 1])

    X_train = df_edge_embeddings_concat.iloc[:, :-1]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, core_dist_n_jobs=-1).fit(X_train)

    threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
    outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
    print(len(outliers))
    outliers

    inliers = np.setdiff1d(np.arange(len(df_edge_embeddings_concat)), outliers)
    print(len(inliers))
    inliers """

    # outlier detection (ECOD)

    # dict_df_edge_embeddings_concat_outlier = {}
    dict_df_edge_embeddings_concat_filter = {}

    for group in tqdm(groups_id):
        df_edge_embeddings_concat = pd.read_csv("{}/output/{}/edge_embeddings/edge-embeddings_concat_{}_{}_{}.csv".format(dir, exp, method, group, option))

        X_train = df_edge_embeddings_concat.iloc[:, 2:-1]

        clf = ECOD()
        clf.fit(X_train)

        X_train["labels"] = clf.labels_ # binary labels (0: inliers, 1: outliers)

        df_edge_embeddings_concat_filter = df_edge_embeddings_concat.copy()
        df_edge_embeddings_concat_filter["labels"] = clf.labels_

        # save
        df_edge_embeddings_concat_filter.to_csv("{}/output/{}/edge_embeddings/edge-embeddings_concat_outlier_{}_{}_{}.csv".format(dir, exp, method, group, option), index=False)
        
        df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter[df_edge_embeddings_concat_filter["labels"] == 0]
        df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter.iloc[:, :-1]

        # dict_df_edge_embeddings_concat_outlier[group] = X_train
        dict_df_edge_embeddings_concat_filter[group] = df_edge_embeddings_concat_filter

    # df_edge_embeddings_concat_filter = dict_df_edge_embeddings_concat_filter[groups_id[0]]

    # plot outliers/inliers
    """ for group in tqdm(groups_id):
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")

        df_aux = dict_df_edge_embeddings_concat_outlier[group]
        print("Total:", len(df_aux))
        
        temp = df_aux[df_aux["labels"] == 0]
        x = temp.iloc[:, 0]
        y = temp.iloc[:, 1]
        z = temp.iloc[:, 2]
        ax.scatter3D(x, y, z, c="red", alpha=0.005)
        print("Num. of inliers:", len(temp))

        temp = df_aux[df_aux["labels"] == 1]
        x = temp.iloc[:, 0]
        y = temp.iloc[:, 1]
        z = temp.iloc[:, 2]
        ax.scatter3D(x, y, z, c="gray", alpha=0.005)
        print("Num. of outliers:", len(temp))

        # show plot
        plt.savefig("{}/output/{}/plots/edge-embeddings_outlier_{}_{}_{}.png".format(dir, exp, method, group, option))
        # plt.show()
        plt.clf() """

    # ###  Filter common edges
    # mapping idx with id
    for group in tqdm(groups_id):
        df_aux = pd.DataFrame(())
        k = 0
        for subgroup in subgroups_id_op[group]:
            df_nodes = pd.read_csv("{}/output/{}/preprocessing/graphs_data/nodes_data_{}_{}.csv".format(dir, exp, group, subgroup))
            dict_id = dict(zip(df_nodes["idx"], df_nodes["id"]))
        
            # mapping
            df_edge_embeddings_concat_filter = dict_df_edge_embeddings_concat_filter[group]
            df_edge_embeddings_concat_filter_aux = df_edge_embeddings_concat_filter[df_edge_embeddings_concat_filter["subgroup"] == k]
        
            # print(df_edge_embeddings_concat_filter)
            df_edge_embeddings_concat_filter_aux["source"] = df_edge_embeddings_concat_filter_aux["source"].map(dict_id)
            df_edge_embeddings_concat_filter_aux["target"] = df_edge_embeddings_concat_filter_aux["target"].map(dict_id)
            df_aux = pd.concat([df_aux, df_edge_embeddings_concat_filter_aux])
            k += 1
        dict_df_edge_embeddings_concat_filter[group] = df_aux

    # format id
    if option:
        for group in tqdm(groups_id):
            # format
            df_edge_embeddings_concat_filter = dict_df_edge_embeddings_concat_filter[group]
            df_edge_embeddings_concat_filter["source"] = df_edge_embeddings_concat_filter["source"].map(lambda x: int(x[1:]))
            df_edge_embeddings_concat_filter["target"] = df_edge_embeddings_concat_filter["target"].map(lambda x: int(x[1:]))

    # filter by different edges
    if option:
        for group in tqdm(groups_id):
            df_edge_embeddings_concat_filter = dict_df_edge_embeddings_concat_filter[group]
            df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter[df_edge_embeddings_concat_filter["source"] != df_edge_embeddings_concat_filter["target"]]
            dict_df_edge_embeddings_concat_filter[group] = df_edge_embeddings_concat_filter

    # count edges and filter by count
    dict_df_edges_filter = {}
    for group in tqdm(groups_id):
        # read
        df_edge_embeddings_concat_filter = dict_df_edge_embeddings_concat_filter[group]
        
        # sort edges
        sort_df_edges(df_edge_embeddings_concat_filter)

        df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter[["source", "target"]].value_counts().to_frame()
        df_edge_embeddings_concat_filter.reset_index(inplace=True)
        df_edge_embeddings_concat_filter.columns = ["source", "target", "count"]
        
        # filter
        df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter[df_edge_embeddings_concat_filter["count"] == len(subgroups_id[group])]
        df_edge_embeddings_concat_filter = df_edge_embeddings_concat_filter.iloc[:, [0, 1]]
        dict_df_edges_filter[group] = df_edge_embeddings_concat_filter

    # ### New correlation
    # read raw data
    df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(dir, exp), index_col=0)
    # df_join_raw.index = df_join_raw.index.astype("str")
    df_join_raw = df_join_raw.iloc[:, 2:]

    # log10
    df_join_raw_log = log10_global(df_join_raw)

    # correlation
    dict_df_corr = {}
    for group in tqdm(groups_id):
        # graph filter
        df_edges_filter = dict_df_edges_filter[group]
        G = nx.from_pandas_edgelist(df_edges_filter.iloc[:, [0, 1]])
        nodes = list(G.nodes())
        
        df_join_raw_filter = df_join_raw_log.loc[nodes, :]
        # df_join_raw_filter = df_join_raw_filter.filter(regex=group, axis=1)
        df_join_raw_filter = df_join_raw_filter.filter(like=group, axis=1)

        df_join_raw_filter_t= df_join_raw_filter.T
        df_join_raw_filter_corr = df_join_raw_filter_t.corr(method="pearson")
        dict_df_corr[group] = df_join_raw_filter_corr

    # get new correlation
    dict_df_edges_filter_weight = {}
    for group in tqdm(groups_id):
        df_edges_filter_weight = dict_df_edges_filter[group].copy()
        df_corr = dict_df_corr[group]

        df_edges_filter_weight["weight"] = df_edges_filter_weight.apply(lambda x: df_corr.loc[x["source"], x["target"]], axis=1)
        df_edges_filter_weight.sort_values(["source", "target"], ascending=True, inplace=True)
        dict_df_edges_filter_weight[group] = df_edges_filter_weight
        # df_edges_filter_weight.to_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}.csv".format(dir, exp, method, group, option), index=False)

    # filter by abs(weight) >= threshold
    for group in tqdm(groups_id):
        df_edges_filter_weight = dict_df_edges_filter_weight[group]
        df_edges_filter_weight_filter = df_edges_filter_weight[df_edges_filter_weight["weight"].abs() >= threshold]
        df_edges_filter_weight_filter.to_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}.csv".format(dir, exp, method, group, option), index=False)
        
    # get weight by subgroups
    # dict_df_edges_filter_weight = get_weight_global(dict_df_edges_filter, exp, groups_id, subgroups_id)

    # ### Filter by STD and average weight
    # dict_df_common_edges = std_global(dict_df_edges_filter_weight, exp, method, groups_id, option, th=0.3, plot=True, save=True)

    # show details
    """ for group in tqdm(groups_id):
        df_common_edges = pd.read_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}.csv".format(dir, exp, method, group, option))
        
        G = nx.from_pandas_edgelist(df_common_edges, "source", "target", edge_attr=["weight"])
        print("Group: {}".format(group))
        graph_detail(G) """