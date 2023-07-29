from django.views.decorators.csrf import csrf_exempt, csrf_protect
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

from utils.response import Resp
from utils.utils import info_graph

from GNN_Unsupervised import experiments as exper

from multiprocessing import Process
import time
import pandas as pd
import networkx as nx
import os

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
        # print(request.data)

        serializer = ExperimentSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.save()

            print(data.id)

            # run experiment
            t1 = Process(target=exper.main, args=(str(data.id), data.raw_data, data.method,
                                                           data.data_variation, data.dimension))
            # starting threads
            t1.start()
            # wait until all threads finish
            # t1.join()

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

                data = []
                for item in files:
                    aux = item.split("_")
                    name = "{}-{}".format(aux[4], aux[5])
                    dir = "{}/GNN_Unsupervised/output/{}/changes/{}".format(os.getcwd(), serializer.data["id"], item)
                    df_change_filter = pd.read_csv(dir)
                    df_change_filter = df_change_filter.iloc[:, [0, 1, 4]]
                    # data[name] = df_change_filter.to_dict(orient="records")

                    # df_temp = df_change_filter["label"].value_counts().to_frame().reset_index()
                    graph = nx.from_pandas_edgelist(df_change_filter, "source", "target", edge_attr=["label"], create_using=nx.DiGraph())
                    
                    values = ['PP', 'Pp', 'PN', 'Pn', 'P?', 'pP', 'pp', 'pN', 'pn', 'p?', 'NP', 'Np', 'NN', 'Nn', 'N?', 'nP', 'np', 'nN', 'nn', 'n?', '?P', '?p', '?N', '?n']
                    labels = []
                    for label in values:
                        labels.append({
                            "label": label,
                            "count": len(df_change_filter[df_change_filter.label == label])
                        })
                    aux_data = {
                        "name": name,
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                        "density": round(nx.density(graph), 4),
                        "labels": labels
                    }
                    data.append(aux_data)
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