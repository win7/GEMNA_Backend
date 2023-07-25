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

## To-Do
- Dynamic edge embeddings operator
- Parallel edge2vec operator (ok)
- Improve mapping idx with id

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