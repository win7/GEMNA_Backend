import os

files = ["a_format_go", "b_prepro_go", "cd_node_edge_go", "e_change_go", "f_biocyc_go"]

for file in files:
    os.system("jupyter nbconvert --to python {}.ipynb".format(file))