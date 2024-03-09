# GNN Unsupervised

### Input format dataset
A file in .csv with follow struture:

| Id | Name | X_1.1 | ... | X_1.n | X_2.1 | ... | X_2.m | X_3.1 | ... | X_3.p | Y_1.1 | ... | Y_1.q | Y_2.1 | ... | Y_2.r | Y_3.1 | ... | Y_3.s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m1 | n1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m2 | n2 |- | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m3 | n3 |- | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m4 | n4 |- | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m5 | n5 |- | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m6 | n6 |- | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| m7 | n7 |- | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| ... | ... | ... | ... | ... |... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Files

1. format_input: format input dataset according to above table (ok)
2. preprocessing: create graph data (nodes, edges) (ok)
3. node-embeddings: generate node-embeddings with DGI or VGAE (ok)
4. processing: generate edge-embeddings and outlier detection (ok)
5. baseline: Greedy algorithm for get maximun common subgraph (ok)
6. comparation: compare baseline (Greedy) with GNN (DGI, VGAE) on data variation
7. change_detection: detect change between correlations
8. processing_biocyc

Aditional Files
1. synthetic_graphs:
2.

Main files:

- 1_format_go.ipynb
- 2_prepo_go.ipynb
- 3_4_node-edge_go.ipynb
- 5_change_go.ipynb
- 6_biocyc_go.ipynb
- baseline.ipynb
- comparation.ipynb

## To-Do
- Dynamic edge embeddings operator
- Parallel edge2vec operator (ok)
- Improve mapping idx with id (ok)
- Test Anomaly detection ()
- Convert graph to line graph
- Improve node features selection (ok)
- First filter by same nodes 

## Notes
- exp2 mutant a
- _go = features

## Notes
- exp1 MS GNN Greedy (experimet 1) (this for manuscript)
- exp2 MS GNN Greedy (experimet 2) (ok)
- exp3 Syntectic GNN Greedy (test)
- exp4 Syntectic GNN Greedy (n=1000, p=0.5)
- exp5 Syntectic GNN Greedy (n=1000, p=0.3) (this for manuscript)
- exp7 improve runtimes
- exp8 Reinhard (1.1, 1.2, ...) (ok)
- exp9 Reinhard (1, 2, 3, 4, ...)
- exp10 Hammerly (1.1, 1.2, ...) D1 concat (ok)
- exp11 Reinhard (1.1, 1.2, ...) without log
- exp12 Hammerly (1.1, 1.2, ...) D2 concat (ok)
- exp13 Reinhard (1.1, 1.2, ...) (for save main files)
- exp14 Reinhard (1.1, 1.2, ...) (ok)
- exp16  Alfredo (1.1, 1.2, ...) (Aligment ID,Average Mz,Metabolite name)

- exp20 Alfredo ok
- exp21 Reinhard ok has normalization = false
- exp22 Reinhard ok has normalization = true
- exp23 Alfredo ok process
- exp11 Single cell
- exp12 Plant

- exp14 new mutants
- exp16 new mutants (with dynamic th corr)

### On colab

!pip install pingouin
!pip install pymp-pypi
!pip install pyod
!pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
!pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html


import sys
import os

from google.colab import drive

drive.mount("/content/drive")

py_file_location = "/content/drive/MyDrive/PUCP/Phd/S4/T4/Code/GNN_Unsupervised"
sys.path.append(os.path.abspath(py_file_location))

%cd /content/drive/MyDrive/PUCP/Phd/S4/T4/Code/GNN_Unsupervised
!pwd

1
!pip install pingouin
!pip install pymp
2
!pip install pingouin
!pip install pymp
3-4
! nvcc --version
!pip install pingouin
!pip install pymp-pypi
!pip install pyod
!pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
!pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
5
!pip install pingouin
!pip install pymp-pypi
!pip install pyod
6
!pip install pingouin
!pip install pymp-pypi

def main(params):
    return exp
def main():

## Import versions
import pandas as pd
import pingouin as pg
import numpy as np
import seaborn as sn
print(pg.__version__) # 0.5.3
print(pd.__version__) # 2.0.3
print(np.__version__) # 1.24.3
print(sn.__version__) # 0.13.0


### Experiments
- exp35, exp39
- exp40, exp44

Mutant
- exp105-109: dim = 3, iter=5   (105)
- exp110-114: dim = 4, iter=5   (110)
- exp115-119: dim = 8, iter=5   (115)
- exp120-124: dim = 16, iter=5  (120)
- exp125-129: dim = 32, iter=5  (125)
- exp130-134: dim = 64, iter=5  (130)
- exp135-139: dim = 128, iter=5 (135)

Reinhard
- exp140-142: dim = 3, iter=3   (140)
- exp143-145: dim = 4, iter=3   (143)
- exp146-148: dim = 8, iter=3   (146)
- exp149-151: dim = 16, iter=3  (149)
- exp152-154: dim = 32, iter=3  (152)
- exp155-157: dim = 64, iter=3 (No)
- exp158-160: dim = 128, iter=3 (No)

Hojas
- exp155-157: dim = 3, iter=3   (155)
- exp158-160: dim = 4, iter=3   (158)
- exp161-163: dim = 8, iter=3   (161)
- exp164-166: dim = 16, iter=3  (164)
- exp167-169: dim = 32, iter=3  (167)
- exp170-172: dim = 64, iter=3  (170)
- exp173-175: dim = 128, iter=3 (173)

Mutant + different edge-embedding
- exp176: dim = 3, iter=1, L1
- exp177: dim = 3, iter=1, L2
- exp178: dim = 3, iter=1, Hadamard
- exp179: dim = 3, iter=1, Average

Mutant + different edge-embedding (copy node-embedding from exp176)
- exp180: dim = 3, iter=1, L1
- exp181: dim = 3, iter=1, L2
- exp182: dim = 3, iter=1, Hadamard
- exp183: dim = 3, iter=1, Average

Tea
- exp184-186: dim = 3, iter=3 (184)

---
Mutant
- exp1: dim = 3
Cancer
- exp2: dim = 3
Leaf
- exp3: dim = 3
Tea
- exp4: dim = 3
Mutant
- exp5: dim = 3, clf=ECOD(contamination=0.1, 0.2)

Experiments with early stopper and contamination=0.05
Mutant
- exp6: dim = 3
Cancer
- exp7: dim = 3
Leaf
- exp8: dim = 3
Tea
- exp9: dim = 3

Mutant (with interactive plot)
- exp10: dim = 3

---
Mutant
- exp 11-13
Cancer
- exp 14-16
Leaf
- exp 17-19

Mutant
- exp20: dim = 3, iter=3  
- exp23: dim = 4, iter=3  
- exp26: dim = 8, iter=3  
- exp29: dim = 16, iter=3 
- exp32: dim = 32, iter=3 
- exp35: dim = 64, iter=3 
- exp38: dim = 128, iter=3

Leaf
- exp41: dim = 3, iter=3 
- exp44: dim = 4, iter=3 
- exp47: dim = 8, iter=3 
- exp50: dim = 16, iter=3
- exp53: dim = 32, iter=3
- exp56: dim = 64, iter=3
- exp59: dim = 128, iter=3

Cancer
- exp62: dim = 3, iter=3  


Reinhard new 1
- exp64: 
Reinhard new 2
- exp65: 

Cancer
- exp66: dim = 3, iter=3  
- exp67: dim = 4, iter=3  
- exp68: dim = 8, iter=3  
- exp71: dim = 16, iter=3 
- exp74: dim = 32, iter=3 
- exp77: dim = 64, iter=3 
- exp80: dim = 128, iter=3