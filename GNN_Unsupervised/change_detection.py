# ### Imports
import os
import sys

dir = os.getcwd() + "/GNN_Unsupervised"
sys.path.append(dir)

from tqdm import tqdm
from utils.utils_v1 import *

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import scipy
import sys

# %load_ext autotime

import json

def count_values(df_column):
    df_count = df_column.value_counts().to_frame().reset_index()
    df_count.loc[len(df_count)] = ["NaN", df_column.isna().sum()]
    df_count.columns = ["value", "count"]
    print("Sum", df_count["count"].sum())
    print(df_count)

def fisher_transform(r):
    # z = 0.5 * (np.log(1 + r) - np.log(1 - r))
    z = 0.5 * np.log((1 + r) / (1 - r))
    return z

def hist(x, th):
    # x = df_change["p_value"]
    plt.hist(x, bins=100)
    plt.axvline(x=th, color="red", lw=1)
    plt.axvline(x=0.5, color="red", lw=1)
    # l = len(df_change) - len(df_change)
    # t = len(df_change)
    # plt.title("Loss: {} of {} ({}%)".format(l, t, round(l*100/t)))
    # plt.savefig("output/{}/plots/common_edges_std_{}_{}_{}.png".format(exp, method, group, option))
    plt.show()
    # plt.clf()

def filter_df_change_ID(df_change_filter, ID):
    df_change_filter_temp = df_change_filter[(df_change_filter["source"] == ID) | (df_change_filter["target"] == ID)]
    return df_change_filter_temp

def main(experiment):
    # ### Parameters
    # Opening JSON file

    file = open("{}/input/parameters_{}.json".format(dir, experiment.id))
    params = json.load(file)

    exp = params["exp"]
    print("Exp:\t\t", exp)

    method = params["method"]
    print("Method:\t\t", method)

    alpha = params["alpha"]
    print("Alpha:\t\t", alpha)

    groups_id = params["groups_id"]
    print("Group id:\t", groups_id)

    subgroups_id = params["subgroups_id"]
    print("Subgroups id:\t", subgroups_id)

    option = params["option"]
    print("Option:\t\t", option)

    control = params["control"]
    controls = control.split(",")
    print("Controls:\t", controls)

    threshold_log2 = params["threshold_log2"]
    print("Threshold log2:\t", threshold_log2)

    for item1 in controls:
        for item2 in groups_id:
            if item1 != item2:
                groups = [item1, item2] # change
                print("Groups:\t\t", groups)
                subgroups = {groups[0]: subgroups_id[groups[0]], groups[1]: subgroups_id[groups[1]]}
                print("Subgroups:\t", subgroups)

                # ### Changes detection

                # #### Read raw data
                df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(dir, exp), index_col=0)        
                # df_join_raw.index = df_join_raw.index.astype("str")

                # #### Read edges
                list_graphs = []
                for k in range(len(groups)):
                    df_edges = pd.read_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}.csv".format(dir, exp, method, groups[k], option))
                    sort_df_edges(df_edges)
                    df_edges.rename(columns={"weight": "weight{}".format(k + 1)}, inplace=True)
                    G = nx.from_pandas_edgelist(df_edges, edge_attr=["weight{}".format(k + 1)])
                    
                    list_graphs.append(G)

                # #### Compose
                R = nx.compose(list_graphs[0], list_graphs[1])

                """ labels = []
                for edge in R.edges():
                    weights = R.get_edge_data(*edge)
                    label = get_label(weights)
                    labels.append(label) """
                nx.set_edge_attributes(R, {(u, v): {"label": get_label(ed, th=0.8)} for u, v, ed in R.edges.data()})

                df_change = nx.to_pandas_edgelist(R)
                df_change = df_change[["source", "target", "weight1", "weight2", "label"]]

                # df_change = df_change.fillna(0)
                # df_change

                # ### Differeces between correlations
                df_change.insert(3, "N1", [len(df_join_raw.filter(like=groups[0]).columns)] * len(df_change))
                df_change.insert(5, "N2", [len(df_join_raw.filter(like=groups[1]).columns)] * len(df_change))

                # differences between correlations
                n1 = len(df_join_raw.filter(like=groups[0]).columns) # len(df_change) - df_change["weight1"].isna().sum() # len(df_change)
                n2 = len(df_join_raw.filter(like=groups[1]).columns) # len(df_change) - df_change["weight2"].isna().sum() # len(df_change)
                print(n1, n2)

                z1 = fisher_transform(df_change["weight1"])
                z2 = fisher_transform(df_change["weight2"])

                sezdiff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
                # print(sezdiff)

                diff = z1 - z2
                ztest = diff / sezdiff

                p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(ztest), loc=0, scale=1))
                df_change["p-value"] = p_value

                # get sigtnificant
                list_significant = []
                for row in df_change.itertuples():
                    # print(row[3], row[5], row[8])
                    if abs(row[3] - row[5]) > 0.1 and row[8] < alpha:
                        list_significant.append("change")
                    elif abs(row[3] - row[5]) < 0.1 and row[8] > 0.5:
                        list_significant.append("stable")
                    elif np.isnan(row[3]) or np.isnan(row[5]):
                        list_significant.append("change*")
                    else:
                        list_significant.append("none")

                # len(list_significant)
                df_change["significant"] = list_significant

                # filter by significant
                # df_change_filter = df_change[df_change["p-value"] < alpha]
                df_change_filter1 = df_change[df_change["significant"] != "none"]

                # save
                df_change_filter1.to_csv("{}/output/{}/changes/changes_edges_significant_{}_{}_{}_{}.csv".format(dir, exp, method, groups[0], groups[1], option), index=False)

                # common subgraph
                G = nx.from_pandas_edgelist(df_change_filter1.iloc[:, [0, 1]])

                # filter raw data by common nodes
                nodes_common = sorted(list(G.nodes()))
                df_join_raw_filter = df_join_raw.loc[nodes_common]

                df_join_raw_filter_log2 = np.log2(df_join_raw_filter.filter(like=groups[1]).mean(axis=1) / df_join_raw_filter.filter(like=groups[0]).mean(axis=1))
                df_join_raw_filter_log2 = df_join_raw_filter_log2.to_frame()
                df_join_raw_filter_log2.columns = ["log2"]

                # filter by log2
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

                # #### Mapping Aligment ID to Average Mz
                df_change_filter = df_change_filter2.copy() # df_change_filter1, df_change_filter2

                dict_aux = df_join_raw.iloc[:, :2].to_dict(orient='dict')
                # dict_aux

                dict_mz = dict_aux["Average Mz"]
                dict_mz = {key: value for key, value in dict_mz.items()}
                # dict_mz

                dict_metabolite = dict_aux["Metabolite name"]
                dict_metabolite = {key: value for key, value in dict_metabolite.items()}

                # mapping
                df_change_filter.insert(len(df_change_filter.columns), "source1", df_change_filter["source"].map(dict_mz))
                df_change_filter.insert(len(df_change_filter.columns), "target1", df_change_filter["target"].map(dict_mz))

                df_change_filter.insert(len(df_change_filter.columns), "source2", df_change_filter["source"].map(dict_metabolite))
                df_change_filter.insert(len(df_change_filter.columns), "target2", df_change_filter["target"].map(dict_metabolite))

                # save
                df_change_filter.to_csv("{}/output/{}/changes/changes_edges_log2_{}_{}_{}_{}.csv".format(dir, exp, method, groups[0], groups[1], option), index=False)

                """ H = nx.from_pandas_edgelist(df_change_filter, "source", "target", edge_attr=["label"], create_using=nx.DiGraph())
                # H.edges(data=True)
                nx.write_gexf(H, "{}/output/{}/changes/changes_edges_p-value_{}_{}_{}_{}.gexf".format(dir, exp, method, groups[0], groups[1], option)) """

                # ### Query
                """ df_temp = count_values(df_change_filter["source"])
                print(df_temp["count"].sum())
                df_temp """

                """ df_temp = count_values(df_change_filter["target"])
                print(df_temp["count"].sum())
                df_temp """

                """ node = "98.05823"
                df_change_filter[(df_change_filter["source"] == node) | (df_change_filter["target"] == node)] """

                # ### Plot
                # HF = H.subgraph(["127.0513", "132.086", "145.0507", "980.0155", "132.086", "115.0038"])
                """ HF = H.subgraph(["173.05193", "139.05135", "98.05823"])
                edge_labels = nx.get_edge_attributes(HF, "label")

                pos = pos=nx.spring_layout(HF)
                nx.draw_networkx(HF, pos, font_color="black", font_size=12, node_color="orange")
                nx.draw_networkx_edge_labels(H, pos, edge_labels, font_size=14)

                # plt.title("{}: {} --> {}".format(method, group1[0], group2[0]))
                plt.show() """