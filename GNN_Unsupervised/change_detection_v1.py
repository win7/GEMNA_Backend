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

def sort_edges(df_edges):
    s = []
    t = []
    for row in df_edges.itertuples():
        if row[1] > row[2]:
            s.append(row[2])
            t.append(row[1])
        else:
            s.append(row[1])
            t.append(row[2])
    df_edges["source"] = s
    df_edges["target"] = t

def count_values(df_column):
    df_count = df_column.value_counts().to_frame().reset_index()
    df_count.loc[len(df_count)] = ["NaN", df_column.isna().sum()]
    df_count.columns = ["value", "count"]
    return df_count

def fisher_transform(r):
    # z = 0.5 * (np.log(1 + r) - np.log(1 - r))
    z = 0.5 * np.log((1 + r) / (1 - r))
    return z

def hist(x, th):
    # x = df_change["p_value"]
    plt.hist(x, bins=100)
    plt.axvline(x=th, color="red", lw=1)
    # l = len(df_change) - len(df_change)
    # t = len(df_change)
    # plt.title("Loss: {} of {} ({}%)".format(l, t, round(l*100/t)))
    # plt.savefig("{}/output/{}/plots/common_edges_std_{}_{}_{}.png".format(dir, exp, method, group, option))
    plt.show()
    # plt.clf()

def main(experiment):
    # ### Parameters
    # Opening JSON file

    file = open("{}/input/parameters_{}.json".format(dir, experiment.id))
    params = json.load(file)

    exp = params["exp"]
    print("Exp:\t\t", exp)

    method = params["method"]
    print("Method:\t\t", method)

    groups_id = params["groups_id"]
    print("Group id:\t", groups_id)

    option = params["option"]
    print("Option:\t\t", option)

    control = params["control"]
    controls = control.split(",")
    print("Controls:\t", controls)

    for item1 in controls:
        for item2 in groups_id:
            if item1 != item2:
                groups = [item1, item2] # change
                print("Groups:\t\t", groups)

                # ### Changes detection

                # #### Read edges
                list_graphs = []
                for k in range(len(groups)):
                    df_edges = pd.read_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}.csv".format(dir, exp, method, groups[k], option),
                                            dtype={"source": "string", "target": "string"})
                    sort_edges(df_edges)
                    df_edges.rename(columns={"weight": "weight{}".format(k + 1)}, inplace=True)
                    G = nx.from_pandas_edgelist(df_edges, edge_attr=["weight{}".format(k + 1)])
                    
                    list_graphs.append(G)

                # #### Compose
                R = nx.compose(list_graphs[0], list_graphs[1])

                labels = []
                for edge in R.edges():
                    weights = R.get_edge_data(*edge)
                    label = get_label(weights)
                    labels.append(label)

                nx.set_edge_attributes(R, {(u, v): {"label": get_label(ed, th=0.8)} for u, v, ed in R.edges.data()})

                df_change = nx.to_pandas_edgelist(R)
                df_change = df_change[["source", "target", "weight1", "weight2", "label"]]

                # df_change = df_change.fillna(0)
                # df_change

                # ### Differences between correlations
                # option 1
                n1 = len(df_change) - df_change["weight1"].isna().sum() # len(df_change)
                n2 = len(df_change) - df_change["weight2"].isna().sum() # len(df_change)

                z1 = fisher_transform(df_change["weight1"])
                z2 = fisher_transform(df_change["weight2"])

                sezdiff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
                # print(sezdiff)

                diff = z1 - z2
                ztest = diff / sezdiff

                p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(ztest), loc=0, scale=1))

                # df_change["z1"] = z1
                # df_change["z2"] = z2
                # df_change["diff"] = diff
                # df_change["ztest"] = ztest
                df_change["p-value"] = p_value

                # df_temp = count_values(df_change["label"])

                # df_temp = count_values(df_change["p-value"])

                """ try:
                    x = df_change["p-value"]
                    hist(x, th=0.05)
                except:
                    pass """

                # filter by p-value
                df_change_filter = df_change.copy() # df_change[df_change["p-value"] < 0.05]

                """ try:
                    x = df_change_filter["p-value"]
                    hist(x, th=0.05)
                except:
                    pass """

                """ df_temp = count_values(df_change_filter["label"])
                print(df_temp["count"].sum())
                df_temp """

                """ df_temp = count_values(df_change_filter["p-value"])
                print(df_temp["count"].sum())
                df_temp """

                # #### Mapping Aligment ID to Average Mz
                """ df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(dir, exp), index_col=0)        
                df_join_raw.index = df_join_raw.index.astype("str")

                dict_aux = df_join_raw.iloc[:, :2].to_dict(orient='dict')
                # dict_aux

                dict_mz = dict_aux["Average Mz"]
                dict_mz = {str(key): value for key, value in dict_mz.items()}
                # dict_mz

                dict_metabolite = dict_aux["Metabolite name"]
                dict_metabolite = {str(key): value for key, value in dict_metabolite.items()}
                # dict_metabolite

                # mapping
                df_change_filter["source"] = df_change_filter["source"].map(dict_mz)
                df_change_filter["target"] = df_change_filter["target"].map(dict_mz) """

                # save
                df_change_filter.to_csv("{}/output/{}/changes/changes_edges_p-value_{}_{}_{}_{}.csv".format(dir, exp, method, groups[0], groups[1], option), index=False)

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