from tqdm import tqdm

import os
import sys

dir = os.getcwd() + "/GNN_Unsupervised"
sys.path.append(dir)

from django.core.mail import EmailMultiAlternatives
from django.db import transaction
from django.conf import settings

from main.models import Experiment
# from .main.serializers import ExperimentSerializer

import a_format_go as fi
import b_prepro_go as pp
import cd_node_edge_go as ne
import e_change_go as ch
import f_biocyc_go as by

import time

def main(experiment):
    # print(exp, raw_data, method, option, dimension, email)
    # exp = "010853ec-3b7f-4255-bfbf-cb36ac59119f"
    print("Start")
    start = time.time()

    print("\nFormat input")
    fi.main(experiment)

    print("\nPreprocessing")
    pp.main(experiment)
    
    print("\nNode-Edge embedding")
    ne.main(experiment)

    print("\nChange detection") 
    ch.main(experiment)

    print("\nProcessing biocyc")
    by.main(experiment)

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
    print(elapsed_time)
    print("End")
    
"""    
if __name__ == "__main__":
    import multiprocessing as mp
    
    id = "1c04eacf-a4cc-4407-8d28-418ed3370c5d"
    experiment = Experiment.objects.get(pk=id)
    
    # mp1.set_start_method('spawn', force=True)
    t1 = mp.Process(target=main, args=(experiment,))
    # starting threads
    t1.start() """