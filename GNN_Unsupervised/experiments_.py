from datetime import timedelta
import time
import os

from tqdm import tqdm

from utils.utils_go import *
import a_format_go as fm
import b_prepro_go as pp
import cd_node_edge_go as ne
# import e_change_go as ch
# import f_biocyc_go as bc

""" files = ["a_format_go", "b_prepro_go", "cd_node_edge_go", "e_change_go", "f_biocyc_go"]

for file in tqdm(files[:1]):
    os.system("python {}.py".format(file)) """
    
params = {
    "methods": ["dgi-tran", "argva-base", "vgae-line", "vgae-base"], # ["deep-walk", "node2vec15", "node2vec12", "dgi-indu", "dgi-tran", "argva-base", "vgae-line", "vgae-base"], # ["vgae", "dgi", "dgi-tran", "dgi-indu", "argva-base", "vgae-line", "vgae-base"]
    "data_variations": ["none", "str", "dyn"],
    "has_transformation": False, # True or False # change
    "control": "FCSglc", # "Nueva", # "Herbal", # "T" # "LG" # "FCSglc" # "WT"
    "dimension": 3,
    "threshold_corr": 0.3, # 0.3 to mutant
    "threshold_log2": 0,
    "alpha": 0.05,
    "iterations": 1,
    "raw_data_file": "reinhard_format", # "tea_format", # "hojas_format", # "reinhard_format" # "mutant_a_format" # "plant" # "single_cell" # ""Trial Alfredo process" # "Trial 1_Reinhard" # "Trial Alfredo" # change
    "obs": "",
    "seeds": [41, 42, 43, 44, 45, 46]
}
print(params)

dimensions = [4] #[3, 4, 8, 16, 32, 64, 128]
for dimension in dimensions:
    params["dimension"] = dimension

    for _ in range(1): # change
        start = time.time()
        exp = fm.main(params)
        pp.main()
        ne.main()
        # ch.main()
        # bc.main()
        end = time.time()
        elapsed = end - start
        td = timedelta(seconds=elapsed)
        # print("Runtime: {}".format(elapsed))
        print("Runtime: {}".format(td))