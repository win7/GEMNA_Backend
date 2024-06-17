from django.views.decorators.csrf import csrf_exempt, csrf_protect
from rest_framework.parsers import FormParser, MultiPartParser
from main.models import Experiment
from main.serializers import ExperimentSerializer
from django.http import Http404
# from django.http import JsonResponse

from django.core.exceptions import ObjectDoesNotExist
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status

# from django.core.mail import EmailMultiAlternatives
from django.db import transaction
from django.conf import settings

from utils.response import Resp
# from utils.utils_v1 import info_graph

import sys
import os
""" path = "/home/ealvarez/Project"
if not path in sys.path:
    sys.path.append(path) """
""" path = "/home/ealvarez/Project/GNN_Unsupervised"
if not path in sys.path:
    sys.path.append(path) """

# Append GNN sources (Important)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join("", "/home/ealvarez/Project"))
sys.path.append(os.path.join("", "/home/ealvarez/Project/GNN_Unsupervised"))
# sys.path.append(os.path.join(BASE_DIR, "/home/ealvarez/Project/GNN_Unsupervised/utils"))
from GNN_Unsupervised import experiments_drf as run_experiment

# from multiprocessing import Process
import multiprocessing as mp
# import threading
# mp.set_start_method('spawn')
# mp.set_start_method('spawn', force=True)
# import torch.multiprocessing as mp1

from sklearn.decomposition import PCA
import pandas as pd
import networkx as nx
import numpy as np

values = ['PP', 'Pp', 'PN', 'Pn', 'P?', 'pP', 'pp', 'pN', 'pn', 'p?', 'NP', 'Np', 'NN', 'Nn', 'N?', 'nP', 'np', 'nN', 'nn', 'n?', '?P', '?p', '?N', '?n']

values_Diff = ['Pp', 'PN', 'Pn', 'P?', 'pP', 'pN', 'pn', 'p?', 'NP', 'Np', 'Nn', 'N?', 'nP', 'np', 'nN', 'n?', '?P', '?p', '?N', '?n']
values_SIM = ['PP', 'pp','NN', 'nn']

exp_path = os.getcwd() + "/experiments"

def log10_global(df_join_raw):
    df_join_raw_log = df_join_raw.copy()
    for column in df_join_raw.columns:
        # df_join_raw_log[column] = np.log10(df_join_raw[column], where=df_join_raw[column]>0)
        df_join_raw_log[column] = np.log10(df_join_raw_log[column])
        df_join_raw_log[column] = df_join_raw_log[column].replace(-np.Inf, np.nan)
        df_join_raw_log[column] = df_join_raw_log[column].replace(np.nan, df_join_raw_log[column].min() / 100)
    return df_join_raw_log

class ExperimentList(APIView):
    """
    List all experiments, or create a new experiment.
    """
    parser_classes = (MultiPartParser, FormParser)
    # permission_classes = (IsAuthenticated,)
	# serializer_class = ExperimentSerialize

    def get(self, request, format=None):
        experiments = Experiment.objects.all()

        serializer = ExperimentSerializer(experiments, many=True)
        return Resp(data=serializer.data, message="Experiments Successfully Recovered.").send()

    def post(self, request, format=None):
        print(request.data)
        
        serializer = ExperimentSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.save()
            # data = Experiment.objects.get(pk="ed539a71-e25a-4dea-80f8-b78572898fd0")

            params = {
                "exp": str(data.id),
                "methods": [data.method], # ["vgae", "dgi", "dgi-tran", "dgi-indu", "argva-base", "vgae-line", "vgae-base"], # ["vgae", "dgi", "dgi-tran", "dgi-indu", "argva-base", "vgae-line", "vgae-base"]
                "data_variations": [data.data_variation], # ["none", "str", "dyn"]
                "has_transformation": data.transformation,
                "controls": data.controls.split(","),
                "dimension": data.dimension,
                "threshold_corr": data.threshold_corr,
                "threshold_log2": data.threshold_log2,
                "alpha": data.alpha,
                "iterations": 1,
                "raw_data_file": data.raw_data.file.name,
                "groups_id_no": ["Blank", "QC", "Std"],
                "obs": "",
                "seeds": [42, 43, 44, 45, 46],
                
                "from": "drf",
                "cuda": 0,
                "epochs": 100,
                "lr": 0.01,
                "weight_decay": 1e-4,
                "patience": 10,
                "contamination": 0.05, # float in (0., 0.5)
                "n_jobs": 1, # -1 all
            }
            print(params)
            
            # run experiment
            """ try:
                mp.set_start_method('fork', force=True)
                print("spawned")
            except RuntimeError:
                print("Error spawn") """
                
            # mp.set_start_method('spawn', force=True)
            # ctx = mp.get_context("spawn")
            
            # mp1.set_start_method('spawn', force=True)
            t1 = mp.Process(target=run_experiment.run, args=(params,))
            t1.start()
            # wait until all threads finish
            # t1.join()
            ## import os
            ##import subprocess
            ##subprocess.check_output(["python3", "/home/ealvarez/Project/MetaNet/GNN_Unsupervised/experiments.py"])
            # os.system("python /home/ealvarez/Project/MetaNet/GNN_Unsupervised/experiments.py")

            return Resp(data=serializer.data, message="Experiment Successfully Created.", status=status.HTTP_201_CREATED).send()
        return Resp(data=serializer.errors, message="Error Creating Experiment.", flag=False, status=status.HTTP_400_BAD_REQUEST).send()

class ExperimentDetail(APIView):
    """
    Retrieve, update or delete a experiment instance.
    """

    # permission_classes = (IsAuthenticated,)
	# serializer_class = ExperimentSerializer

    def get_object(self, pk):
        try:
            experiment = Experiment.objects.get(pk=pk)
            return experiment
        except Experiment.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        try:
            try:
                print(request.GET)
                f = request.GET["quality"]
                
                experiment = Experiment.objects.get(pk=pk)
                serializer = ExperimentSerializer(experiment)
                
                files = os.listdir("{}/output/{}/changes/".format(exp_path, serializer.data["id"]))
                files.sort()

                details = []
                nodes = {}

                # load data
                df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(exp_path, experiment.id), index_col=0) #, usecols=[0, 1, 2])
                # df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(exp_path, experiment.id), index_col=0, usecols=[0, 1, 2])
                # df_join_raw.index = df_join_raw.index.astype("str")
                # df_join_raw.columns = ["mz", "name"]
                df_join_raw.rename(columns={"Average Rt": "rt", "Average Mz": "mz", "Metabolite name": "name"}, inplace=True)
                # print(df_join_raw)
                # print(files)
                
                # get files names
                file_names = []
                for item in files:
                    if "log2_{}".format(f) in item:
                        file_names.append(item)
                file_names.sort()
                print(file_names)
                    
                for item in file_names:
                    aux = item.split("_")
                    name = "{}-{}".format(aux[5], aux[6]) # important
                    # print(item)
                    df_change_filter = pd.read_csv("{}/output/{}/changes/{}".format(exp_path, serializer.data["id"], item)) # , dtype={"source": "string", "target": "string"})
                    df_change_filter = df_change_filter[((df_change_filter["significant"] == "*")) | 
                                                    ((df_change_filter["significant"] == "-") & (df_change_filter["label"].str[0] == df_change_filter["label"].str[1]))]
                    # print(df_change_filter)
                    # df_change_filter = df_change_filter.iloc[:, [0, 1, 6]]
                    graph = nx.from_pandas_edgelist(df_change_filter.iloc[:, [0, 1, 6]], "source", "target", edge_attr=["label"]) #, create_using=nx.DiGraph())
                    # nodes += list(graph.nodes())
                    
                    labels = []
                    for label in values:
                        labels.append({
                            # "label": label,
                            "count": len(df_change_filter[df_change_filter.label == label])
                        })
                    aux_data = {
                        "name": name,
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                        "density": round(nx.density(graph), 4),
                        "labels": labels
                    }
                    details.append(aux_data)
                    
                    ids = np.unique(df_change_filter.iloc[:, [0, 1]].values.flatten())

                    df_nodes = df_join_raw.loc[ids] # ["Average Mz", "Metabolite name"]
                    df_nodes.insert(0, "id", df_nodes.index)
                    # print(df_nodes)
                    nodes[name] = df_nodes.iloc[:, :4].to_dict(orient="records")
                # nodes = np.unique(nodes)
                
                # clustering
                # print(serializer.data)
                exp = serializer.data["id"]
                method =serializer.data["method"]
                group_id = serializer.data["controls"].split(",")[0]
                data_variation =  serializer.data["data_variation"]
                iteration = 1
                # f = "f1"
                print(exp, method, group_id, data_variation)
                
                df_edges_filter_weight_filter = pd.read_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}_{}.csv".format(exp_path,
                                                                                                                            exp,
                                                                                                                            f,
                                                                                                                            method,
                                                                                                                            group_id,
                                                                                                                            data_variation))
                
                nodes_ = np.unique(df_edges_filter_weight_filter.iloc[:, [0, 1]].values.flatten())
                df_join_raw_filter = df_join_raw.loc[nodes_]
                # print(df_join_raw_filter)
                
                X = df_join_raw_filter.iloc[:, 3:]
                X = log10_global(X)
                X = X.T

                # colors = ["", "white", "green", "black", "yellow", "red", "orange"]

                classes = [x.split("_")[0] for x in X.index]
                # print(len(classes), classes)

                pca = PCA()
                pca.fit(X)
                X = pca.transform(X)

                df_cluster = pd.DataFrame(X)
                df_cluster["Class"] = classes
                df_cluster = df_cluster.iloc[:, [0, 1, -1]]
                # print(df_cluster)

                data = {
                    "details": details,
                    "nodes": nodes,
                    # "cluster": df_cluster.values
                }
                return Resp(data=data, message="Experiment Successfully Recovered.").send()
            except Exception as e:
                print(str(e))
                return Resp(message="Experiment Does Not Exist.", flag=False, status=status.HTTP_500_INTERNAL_SERVER_ERROR).send()
        except Experiment.DoesNotExist:
            return Resp(message="Experiment Does Not Exist.", flag=False, status=status.HTTP_404_NOT_FOUND).send()
        
    def put(self, request, pk, format=None):
        experiment = self.get_object(pk)
        serializer = ExperimentSerializer(experiment, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Resp(data=serializer.data, message="Experiment Updated Successfully.").send()
        return Resp(data=serializer.errors, message="Error Updating Experiment.", flag=False, status=status.HTTP_400_BAD_REQUEST).send()

    def delete(self, request, pk, format=None):
        experiment = self.get_object(pk)
        experiment.delete()
        return Resp(message="Experiment Deleted Successfully.", status=status.HTTP_204_NO_CONTENT).send()

class ExperimentConsult(APIView):
    """
    Consult experiment.
    """
    # permission_classes = (IsAuthenticated,)
	# serializer_class = ExperimentSerialize

    def get(self, request, format=None):
        return Resp(message="", status=status.HTTP_501_NOT_IMPLEMENTED).send()

    def post(self, request, format=None):
        try:
            try:
                print(request.data)

                pk = request.data["id"]
                groups = request.data["groups"]
                group = request.data["group"]
                nodes = request.data["nodes"]
                type = request.data["type"]
                plot = request.data["plot"]
                f = request.data["quality"]

                experiment = Experiment.objects.get(pk=pk)

                df_change_filter = pd.read_csv("{}/output/{}/changes/changes_edges_log2_{}_{}_{}_{}.csv".format(exp_path,
                                                                                                                experiment.pk,
                                                                                                                f,
                                                                                                                experiment.method,
                                                                                                                group.replace("-", "_"),
                                                                                                                experiment.data_variation),
                                                dtype={"source": "string", "target": "string", "source1": "string", "target1": "string"})
                # df_change_filter = pd.read_csv(dir1)
            
                print(df_change_filter)
                # df_change_filter = df_change_filter.iloc[:, [0, 1, 4]]
                # print(df_change_filter.iloc[:20,:])
                
                # Filter by significant correlations
                df_change_filter = df_change_filter[((df_change_filter["significant"] == "*")) | 
                                                    ((df_change_filter["significant"] == "-") & (df_change_filter["label"].str[0] == df_change_filter["label"].str[1]))]
                
                # print(df_change_filter.iloc[:, key_subgraph[type]])
                # print(list(df_change_filter.iloc[:, key_subgraph[type]]))
                H = nx.from_pandas_edgelist(df_change_filter.iloc[:, [0, 1, 6]], "source", "target", # *df_change_filter.iloc[:, key_subgraph[type][:2]].columns, 
                                            edge_attr=["label"]) # , create_using=nx.DiGraph())
                
                if "-1" in nodes: # get all nodes
                    nodes = list(H.nodes())
                    # print(nodes)
                    
                    HF = H.subgraph(nodes)
                else:
                    # get neighbors                
                    if plot == "correlation_neighbors":
                        # option 1: 
                        """ aux_nodes = nodes.copy()
                        for node in aux_nodes:
                            nodes += list(H.neighbors(node)) """
                        
                        # option 1: 
                        list_graph = []
                        aux_nodes = nodes.copy()
                        for node in aux_nodes:
                            H_ = nx.ego_graph(H, node)
                            list_graph.append(H_)
                            
                        C = nx.compose_all(list_graph)
                        
                        # delete edges
                        edges = list(C.edges())
                        for edge in edges:
                            if not edge[0] in nodes and not edge[1] in nodes:
                                C.remove_edge(*edge)
                        
                        HF = C.copy() # H.subgraph(nodes) # H.subgraph(nodes) or H # graph or subgraph or C (compose graph)
                    else:
                        HF = H.subgraph(nodes)
                    
                df_change_filter_sub = nx.to_pandas_edgelist(HF)
                pos = nx.spring_layout(HF, seed=42)
                nx.set_node_attributes(HF, pos, "pos")

                # Degrees
                new_nodes = list(HF.nodes())
                print(new_nodes)
                # degrees = sorted(H.degree, key=lambda x: x[1], reverse=True)
                # degrees = np.array([[int(node), val] for (node, val) in HF.degree()]) # H.degree (all), HF.degree (part)
                degrees = np.array(list(H.degree(new_nodes))) # before
                # degrees = degrees[degrees[:, 0].argsort()]

                # edge_labels = nx.get_edge_attributes(HF, "label")
                # print(df_change_filter.to_dict(orient="list"))
                """ df_change_filter_temp = df_change_filter.iloc[:500, :]

                all_nodes = pd.unique(pd.concat([df_change_filter_temp['source'], df_change_filter_temp['target']]))
                matrix = pd.DataFrame(0, index=all_nodes, columns=all_nodes)

                # Fill the matrix with 1 where there are edges
                for _, row in df_change_filter_temp.iterrows():
                    matrix.loc[row['source'], row['target']] = row['weight1'] - row['weight2']
                    matrix.loc[row['target'], row['source']] = row['weight2'] - row['weight1'] """

                # BioCyc
                groups.remove(group)
                groups.insert(0, group)
                dict_biocyc = {}
                
                for group_ in groups:
                    df_biocyc = pd.read_csv("{}/output/{}/biocyc/biocyc_{}_{}_{}_{}.csv".format(exp_path,
                                                                                                experiment.pk,
                                                                                                f,
                                                                                                experiment.method, 
                                                                                                group_, 
                                                                                                experiment.data_variation), 
                                                                                            delimiter="\t",
                                                                                            dtype={"Alignment ID": "string"}) # , names=["name", "mz", "id", "Before", "After", "Ratio"])
                    df_biocyc.columns = ["name", "mz", "id", "Before", "After", "Ratio"]
                    # df_biocyc = df_biocyc.loc[:, [type, "Before", "After", "Ratio"]]
                    # df_biocyc = df_biocyc.round(2)
                    # df_biocyc.columns = ["ID", "Before", "After", "Ratio"]
                    # df_biocyc.fillna(0, inplace=True)
                    list_temp = []
                    for k, node in enumerate(new_nodes):
                        df_temp = df_biocyc[df_biocyc["id"] == node]
                        # print(group_, df_temp)
                        if len(df_temp) > 0:
                            df_temp = df_temp.loc[:, ["id", "Before", "After", "Ratio"]] # [type, "Before", "After", "Ratio"]
                            # df_temp.columns = ["ID", "Before", "After", "Ratio"]
                            list_temp.append(df_temp.to_dict(orient="records")[0])
                        else:
                            # list_temp.append({"ID": dict_biocyc[group][k]["ID"], "Before": 0, "After": 0, "Ratio": 0})
                            list_temp.append({"id": dict_biocyc[group][k]["id"], "Before": "-", "After": "-", "Ratio": "-"})
                    dict_biocyc[group_] = list_temp
                    """ df_biocyc = df_biocyc[df_biocyc["id"].isin(list(map(int, nodes)))] # filter from part of dataframe
                    # print(df_biocyc.info())
                    df_biocyc.sort_values(by=["id"], inplace=True) # sort_values(by=[type], inplace=True)
                    # print(df_biocyc)
                    df_biocyc = df_biocyc.loc[:, [type, "Before", "After", "Ratio"]]
                    # df_biocyc = df_biocyc.round(2)
                    df_biocyc.columns = ["ID", "Before", "After", "Ratio"]                   
                    df_biocyc.fillna(0, inplace=True)
                    dict_biocyc[group_] = df_biocyc.to_dict(orient="records") """
                # print(dict_biocyc)

                data = {
                    # "changes": matrix.to_dict(orient="list"), # df_change_filter.to_dict(orient="records"),
                    "nodes": [{"id": str(node), **data} for node, data in HF.nodes(data=True)],
                    "edges": df_change_filter_sub.to_dict(orient="records"),
                    "degrees": degrees,
                    # "biocyc": dict_biocyc[group],
                    "biocyc_all": dict_biocyc
                    # "edges": df_change_filter_sub.to_dict(orient="records"),
                }
                # print(data)
                print("End...")
                return Resp(data=data, message="Experiment Successfully Recovered.").send()
            except Exception as e:
                print(str(e))
                return Resp(message=str(e), flag=False, status=status.HTTP_500_INTERNAL_SERVER_ERROR).send()
        except Experiment.DoesNotExist:
            return Resp(message="Experiment Does Not Exist.", flag=False, status=status.HTTP_404_NOT_FOUND).send()

class ExperimentFinetune(APIView):
    """
    Retrieve, update or delete a experiment instance.
    """

    # permission_classes = (IsAuthenticated,)
	# serializer_class = ExperimentSerializer

    def get_object(self, pk):
        try:
            experiment = Experiment.objects.get(pk=pk)
            return experiment
        except Experiment.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        return Resp(message="", status=status.HTTP_501_NOT_IMPLEMENTED).send()
        
    def post(self, request, format=None):
        try:
            try:
                print(request.GET)
                f = request.GET["quality"]
                
                experiment = Experiment.objects.get(pk=pk)
                serializer = ExperimentSerializer(experiment)
                
                files = os.listdir("{}/output/{}/changes/".format(exp_path, serializer.data["id"]))
                files.sort()

                details = []
                nodes = {}

                # load data
                df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(exp_path, experiment.id), index_col=0) #, usecols=[0, 1, 2])
                # df_join_raw = pd.read_csv("{}/input/{}_raw.csv".format(exp_path, experiment.id), index_col=0, usecols=[0, 1, 2])
                # df_join_raw.index = df_join_raw.index.astype("str")
                # df_join_raw.columns = ["mz", "name"]
                df_join_raw.rename(columns={"Average Mz": "mz", "Metabolite name": "name"}, inplace=True)
                # print(df_join_raw)
                # print(files)
                
                # get files names
                file_names = []
                for item in files:
                    if "significant" in item or "compose" in item or "summary" in item or f not in item:
                        continue
                    file_names.append(item)
                file_names.sort()
                # print(file_names)
                    
                for item in file_names:
                    aux = item.split("_")
                    name = "{}-{}".format(aux[5], aux[6]) # important
                    # print(item)
                    df_change_filter = pd.read_csv("{}/output/{}/changes/{}".format(exp_path, serializer.data["id"], item)) # , dtype={"source": "string", "target": "string"})
                    df_change_filter = df_change_filter[((df_change_filter["significant"] == "*")) | 
                                                    ((df_change_filter["significant"] == "-") & (df_change_filter["label"].str[0] == df_change_filter["label"].str[1]))]
                    # print(df_change_filter)
                    # df_change_filter = df_change_filter.iloc[:, [0, 1, 6]]
                    graph = nx.from_pandas_edgelist(df_change_filter.iloc[:, [0, 1, 6]], "source", "target", edge_attr=["label"]) #, create_using=nx.DiGraph())
                    # nodes += list(graph.nodes())
                    
                    labels = []
                    for label in values:
                        labels.append({
                            # "label": label,
                            "count": len(df_change_filter[df_change_filter.label == label])
                        })
                    aux_data = {
                        "name": name,
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                        "density": round(nx.density(graph), 4),
                        "labels": labels
                    }
                    details.append(aux_data)
                    
                    ids = np.unique(df_change_filter.iloc[:, [0, 1]].values.flatten())

                    df_nodes = df_join_raw.loc[ids] # ["Average Mz", "Metabolite name"]
                    df_nodes.insert(0, "id", df_nodes.index)
                    # print(df_nodes)
                    nodes[name] = df_nodes.iloc[:, :4].to_dict(orient="records")
                # nodes = np.unique(nodes)
                
                # clustering
                # print(serializer.data)
                exp = serializer.data["id"]
                method =serializer.data["method"]
                group_id = serializer.data["controls"].split(",")[0]
                data_variation =  serializer.data["data_variation"]
                iteration = 1
                # f = "f1"
                print(exp, method, group_id, data_variation)
                
                df_edges_filter_weight_filter = pd.read_csv("{}/output/{}/common_edges/common_edges_{}_{}_{}_{}.csv".format(exp_path,
                                                                                                                            exp,
                                                                                                                            f,
                                                                                                                            method,
                                                                                                                            group_id,
                                                                                                                            data_variation))
                
                nodes_ = np.unique(df_edges_filter_weight_filter.iloc[:, [0, 1]].values.flatten())
                df_join_raw_filter = df_join_raw.loc[nodes_]
                # print(df_join_raw_filter)
                
                X = df_join_raw_filter.iloc[:, 3:]
                X = log10_global(X)
                X = X.T

                # colors = ["", "white", "green", "black", "yellow", "red", "orange"]

                classes = [x.split("_")[0] for x in X.index]
                # print(len(classes), classes)

                pca = PCA()
                pca.fit(X)
                X = pca.transform(X)

                df_cluster = pd.DataFrame(X)
                df_cluster["Class"] = classes
                df_cluster = df_cluster.iloc[:, [0, 1, -1]]
                # print(df_cluster)

                data = {
                    "details": details,
                    "nodes": nodes,
                    # "cluster": df_cluster.values
                }
                return Resp(data=data, message="Experiment Successfully Recovered.").send()
            except Exception as e:
                print(str(e))
                return Resp(message="Experiment Does Not Exist.", flag=False, status=status.HTTP_500_INTERNAL_SERVER_ERROR).send()
        except Experiment.DoesNotExist:
            return Resp(message="Experiment Does Not Exist.", flag=False, status=status.HTTP_404_NOT_FOUND).send()
