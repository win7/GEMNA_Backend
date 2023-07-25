import os
import sys

print(sys.path)
sys.path.append("/home/ealvarez/Project/MetaNet/GNN_Unsupervised/")

import format_input as fi
import preprocessing as pp
import node_embeddings_dgi as dgi
import processing as pg
import change_detection as cd
import processing_biocyc as pb

import time

# Preprocessing

# Node-embeddings:
# methods = ["dgi", "vgae"]
# options = ["", "dyn", "str"]


# 
# os.system("python del.py 123 456")
# os.system("jupyter nbconvert --execute --to notebook --inplace z.ipynb")

""" exp = "exp102"
method = "dgi" # vgae, dgi
dimension = 3
option = "dyn" # dyn, str
raw_data_file = "Trial 1_Reinhard" """

def main(exp, raw_data, method, option, dimension):
    # exp = "22aac3c6-cc96-4fc7-be5f-ffe04ae19d86"
    print("Start")
    start = time.time()

    fi.main(exp, raw_data, method, option, dimension)

    pp.main(exp)

    dgi.main(exp)

    pg.main(exp)

    cd.main(exp)

    pb.main(exp)

    end = time.time()
    elapsed_time = round((end - start) / 60, 2)
    print(elapsed_time)
    print("End")