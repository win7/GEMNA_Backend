from tqdm import tqdm

import os
import sys

# dir = os.getcwd() + "/GNN_Unsupervised"
# sys.path.append(dir)

from django.core.mail import EmailMultiAlternatives
from django.db import transaction
from django.conf import settings

from main.models import Experiment
from main.serializers import ExperimentSerializer

import format_go as fi
import prepro_go as pp
import node_edge_go as ne
""" import processing as pg
import change_detection as cd
import processing_biocyc as pb """

import time

""" exp = "518b4e5d-4097-486a-907b-936002091f5e"
raw_data = "GNN_Unsupervised/raw_data/Trial_1_Reinhard_0hXQP9m.csv"
method = "dgi" # vgae, dgi
dimension = 3
option = "dyn" # dyn, str
email = "win7.eam@gmail.com" """

def main(experiment):
    # print(exp, raw_data, method, option, dimension, email)
    # exp = "010853ec-3b7f-4255-bfbf-cb36ac59119f"
    print("Start")
    start = time.time()
    
    """ files = ["1_format_go", "2_prepro_go"] # , "3-4_node-edge_go", "5_change_go", "6_biocyc_go"]

    for file in tqdm(files):
        os.system("python GNN_Unsupervised/{}.py".format(file)) """

    print("\nFormat input")
    fi.main(experiment)

    print("\nPreprocessing")
    pp.main(experiment)
    
    print("\nNode-Edge embedding")
    ne.main(experiment)

    """ print("\nNode embeddings")
    dgi.main(experiment)

    print("\nProcessing")
    pg.main(experiment)

    print("\nChange detection") 
    cd.main(experiment)

    print("\nProcessing biocyc")
    pb.main(experiment)

    subject, from_email, to = 'Metabolomic Analysis', 'edwin.alvarez@pucp.edu.pe', experiment.email
    text_content = 'This is an important message.'
    html_content = '<p>This is the experiment code <strong>' + str(experiment.id) + '</strong>.</p>'
    msg = EmailMultiAlternatives(subject, text_content, from_email, [to])
    msg.attach_alternative(html_content, "text/html")

    dir = "{}/GNN_Unsupervised/output/{}/biocyc/".format(os.getcwd(), experiment.id)

    files = os.listdir(dir)
    for item in files:
        aux_dir = "{}{}".format(dir, item)
        with open(aux_dir) as file:
            msg.attach(item, file.read(), "application/csv")
    msg.send()

    end = time.time()
    elapsed_time = round((end - start) / 60, 2)

    # experiment = Experiment.objects.get(pk=experiment.id)
    experiment.runtime = elapsed_time
    experiment.save()
    print(elapsed_time) """
    print("End")