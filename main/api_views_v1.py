mfrom django.views.decorators.csrf import csrf_exempt, csrf_protect
from rest_framework.parsers import FormParser, MultiPartParser
from main.models import Experiment
from main.serializers import ExperimentSerializer
from django.http import Http404
from django.http import JsonResponse

from django.core.exceptions import ObjectDoesNotExist
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status

from django.core.mail import EmailMultiAlternatives
from django.db import transaction
from django.conf import settings

from utils.response import Resp
# from utils.utils_v1 import info_graph

from GNN_Unsupervised import experiments as exper

# from multiprocessing import Process
import multiprocessing as mp
# import threading
# mp.set_start_method('spawn')
# mp.set_start_method('spawn', force=True)
import torch.multiprocessing as mp1

import time
import pandas as pd
import networkx as nx
import numpy as np
import os

values = ['PP', 'Pp', 'PN', 'Pn', 'P?', 'pP', 'pp', 'pN', 'pn', 'p?', 'NP', 'Np', 'NN', 'Nn', 'N?', 'nP', 'np', 'nN', 'nn', 'n?', '?P', '?p', '?N', '?n']

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
            # data = Experiment.objects.get(pk="f02f05f8-bdce-4511-a746-a1e680da9e19")

            # run experiment
            """ try:
                mp.set_start_method('fork', force=True)
                print("spawned")
            except RuntimeError:
                print("Error spawn") """
                
            # mp.set_start_method('spawn', force=True)
            # ctx = mp.get_context("spawn")
            
            # mp1.set_start_method('spawn', force=True)
            t1 = mp1.Process(target=exper.main, args=(data,))
            # starting threads
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
                experiment = Experiment.objects.get(pk=pk)
                serializer = ExperimentSerializer(experiment)
                
                dir_path = "{}/GNN_Unsupervised/output/{}/changes/".format(os.getcwd(), serializer.data["id"])
                files = os.listdir(dir_path)

                details = []
                nodes = {}

                # load data
                df_join_raw = pd.read_csv("{}/GNN_Unsupervised/input/{}_raw.csv".format(os.getcwd(), experiment.id), index_col=0, usecols=[0, 1, 2])        
                # df_join_raw.index = df_join_raw.index.astype("str")
                df_join_raw.columns = ["mz", "name"]
                # print(df_join_raw.head(10))

                for item in files:
                    if "significant" in item:
                        continue
                    aux = item.split("_")
                    name = "{}-{}".format(aux[4], aux[5])
                    print(name)
                    dir = "{}/GNN_Unsupervised/output/{}/changes/{}".format(os.getcwd(), serializer.data["id"], item)
                    df_change_filter = pd.read_csv(dir) # , dtype={"source": "string", "target": "string"})
                    print(df_change_filter)
                    # df_change_filter = df_change_filter.iloc[:, [0, 1, 4]]
                    graph = nx.from_pandas_edgelist(df_change_filter.iloc[:, [0, 1, 6]], "source", "target", edge_attr=["label"], create_using=nx.DiGraph())
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
                    print(df_nodes)
                    nodes[name] = df_nodes.to_dict(orient="records"), 
                    """ {
                        "id": df_change_filter.iloc[:, [0, 1]].values.flatten(),
                        "mz": df_change_filter.iloc[:, [5, 6]].values.flatten(),
                        "name": df_change_filter.iloc[:, [7, 8]].values.flatten()
                    } """
                # nodes = np.unique(nodes)
                data = {
                    "details": details,
                    "nodes": nodes
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
                group = request.data["group"]
                nodes = request.data["nodes"]
                type = request.data["type"]

                experiment = Experiment.objects.get(pk=pk)

                dir1 = "{}/GNN_Unsupervised/output/{}/changes/changes_edges_log2_{}_{}_{}.csv".format(os.getcwd(), experiment.pk, 
                                                                                                         experiment.method, group.replace("-", "_"), experiment.data_variation)
                df_change_filter = pd.read_csv(dir1, dtype={"source": "string", "target": "string",
                                                            "source1": "string", "target1": "string"})
                # df_change_filter = pd.read_csv(dir1)
            
                print(df_change_filter)
                # df_change_filter = df_change_filter.iloc[:, [0, 1, 4]]
                # print(df_change_filter.iloc[:20,:])

                """ key_subgraph = {
                    "id": [0, 1, 4],
                    "mz": [5, 6, 4],
                    "name": [7, 8, 4]
                } """

                # res = df_change_filter[df_change_filter.isin(nodes)]
                # print(res)

                # print(df_change_filter.iloc[:, key_subgraph[type]])
                # print(list(df_change_filter.iloc[:, key_subgraph[type]]))
                H = nx.from_pandas_edgelist(df_change_filter.iloc[:, [0, 1, 6]], "source", "target", # *df_change_filter.iloc[:, key_subgraph[type][:2]].columns, 
                                            edge_attr=["label"], create_using=nx.DiGraph())
                HF = H.subgraph(nodes) # H.subgraph(nodes) or H # graph or subgraph
                df_change_filter_sub = nx.to_pandas_edgelist(HF)
                # print(df_change_filter_sub)

                # degrees = sorted(H.degree, key=lambda x: x[1], reverse=True)
                degrees = np.array([[int(node), val] for (node, val) in H.degree()])
                degrees = degrees[degrees[:, 0].argsort()]
                # print(degrees)

                # edge_labels = nx.get_edge_attributes(HF, "label")
                # print(df_change_filter.to_dict(orient="list"))
                """ df_change_filter_temp = df_change_filter.iloc[:500, :]

                all_nodes = pd.unique(pd.concat([df_change_filter_temp['source'], df_change_filter_temp['target']]))
                matrix = pd.DataFrame(0, index=all_nodes, columns=all_nodes)

                # Fill the matrix with 1 where there are edges
                for _, row in df_change_filter_temp.iterrows():
                    matrix.loc[row['source'], row['target']] = row['weight1'] - row['weight2']
                    matrix.loc[row['target'], row['source']] = row['weight2'] - row['weight1'] """

                dir2 = "{}/GNN_Unsupervised/output/{}/biocyc/biocyc_{}_{}_{}.csv".format(os.getcwd(), experiment.pk, 
                                                                                         experiment.method, group, experiment.data_variation)
                df_biocyc = pd.read_csv(dir2, delimiter="\t", names=["name", "mz", "id", "Before", "After", "Ratio"])
                # print(df_biocyc.info())
                df_biocyc.sort_values(by=["id"], inplace=True) # sort_values(by=[type], inplace=True)
                print(df_biocyc)
                df_biocyc = df_biocyc.loc[:, [type, "Before", "After"]]
                df_biocyc.columns = ["ID", "Before", "After"]
                print(df_biocyc)
                # print(df_biocyc.info())

                data = {
                    # "changes": matrix.to_dict(orient="list"), # df_change_filter.to_dict(orient="records"),
                    "changes_sub": df_change_filter_sub.to_dict(orient="records"),
                    "biocyc": df_biocyc.to_dict(orient="records"),
                    "degrees": degrees
                }
                return Resp(data=data, message="Experiment Successfully Recovered.").send()
            except Exception as e:
                print(str(e))
                return Resp(message="Experiment Does Not Exist.", flag=False, status=status.HTTP_500_INTERNAL_SERVER_ERROR).send()
        except Experiment.DoesNotExist:
            return Resp(message="Experiment Does Not Exist.", flag=False, status=status.HTTP_404_NOT_FOUND).send()
