import os
import sys

dir = os.getcwd() + "/GNN_Unsupervised"
sys.path.append(dir)

from django.core.mail import EmailMultiAlternatives
from django.db import transaction
from django.conf import settings

import format_input as fi
import preprocessing as pp
import node_embeddings_dgi as dgi
import processing as pg
import change_detection as cd
import processing_biocyc as pb

import time

""" exp = "exp102"
method = "dgi" # vgae, dgi
dimension = 3
option = "dyn" # dyn, str
raw_data_file = "Trial 1_Reinhard" """

def main(exp, raw_data, method, option, dimension, email):
    # exp = "83d59431-f3f2-4258-8cd0-5124026e4e5c"
    print("Start")
    start = time.time()

    fi.main(exp, raw_data, method, option, dimension)

    pp.main(exp)

    dgi.main(exp)

    pg.main(exp)

    cd.main(exp)

    pb.main(exp)

    """ email = "edwin.alvarez@pucp.edu.pe"
    subject, from_email, to = "Confirmación de Pago", settings.EMAIL_HOST_USER, email
    text_content = "Gracias..."
    html_content = "<h1>Test</>"

    msg = EmailMultiAlternatives(subject, text_content, from_email, [to])
    msg.attach_alternative(html_content, "text/html")
    msg.send() """

    subject, from_email, to = 'Metabolomic Analysis', 'edwin.alvarez@pucp.edu.pe', email
    text_content = 'This is an important message.'
    html_content = '<p>This is the experiment code <strong>' + exp + '</strong>.</p>'
    msg = EmailMultiAlternatives(subject, text_content, from_email, [to])
    msg.attach_alternative(html_content, "text/html")

    dir = "{}/GNN_Unsupervised/output/{}/biocyc/".format(os.getcwd(), exp)

    files = os.listdir(dir)
    for item in files:
        aux_dir = "{}{}".format(dir, item)
        with open(aux_dir) as file:
            msg.attach(item, file.read(), "application/csv")
    msg.send()

    end = time.time()
    elapsed_time = round((end - start) / 60, 2)
    print(elapsed_time)
    print("End")