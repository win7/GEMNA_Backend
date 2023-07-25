# %% [markdown]
# ### Imports

# %%
import os
import sys

print(sys.path)
sys.path.append("/home/ealvarez/Project/MetaNet/GNN_Unsupervised/utils")

from tqdm import tqdm
from utils import *

import argparse, time

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgi.dgi import DGI
from dgl import DGLGraph
from dgl.data import load_data, register_data_args, DGLDataset

import os

# %%
from tqdm import tqdm
import pandas as pd

os.environ["DGLBACKEND"] = "pytorch"
# %load_ext autotime

# %%
import sys

dir = os.getcwd() + "/GNN_Unsupervised"
sys.path.append(dir)

# from utils.utils import *

# %%
torch.manual_seed(42)
np.random.seed(42)

# %%

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

class CustomDataset(DGLDataset):
    def __init__(self, name, nodes_data, edges_data):
        self.dir = dir
        self.nodes_data = nodes_data
        self.edges_data = edges_data
        super().__init__(name=name)
    
    def process(self):
        node_features = torch.from_numpy(self.nodes_data["degree"].to_numpy())
        # node_features = torch.from_numpy(np.log10(self.nodes_data["degree"].to_numpy()))
        node_features = node_features.to(torch.float32)
        node_features = torch.reshape(node_features, (-1, 1))

        # node_labels = torch.from_numpy(self.nodes_data["id"].to_numpy())
        # node_labels = node_labels.to(torch.float32)

        edge_features = torch.from_numpy(self.edges_data["weight"].to_numpy())
        edges_src = torch.from_numpy(self.edges_data["source"].to_numpy())
        edges_dst = torch.from_numpy(self.edges_data["target"].to_numpy())

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=self.nodes_data.shape[0]
        )
        self.graph.ndata["feat"] = node_features
        # self.graph.ndata["label"] = node_labels
        self.graph.edata["weight"] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = self.nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

def train_dgi(exp, graph, args, method, group, subgroup):
    features = torch.FloatTensor(np.log10(graph.ndata["feat"]))
    # print(features.shape)
    # labels = torch.LongTensor(graph.ndata["label"])
    if hasattr(torch, "BoolTensor"):
        train_mask = torch.BoolTensor(graph.ndata["train_mask"])
        val_mask = torch.BoolTensor(graph.ndata["val_mask"])
        test_mask = torch.BoolTensor(graph.ndata["test_mask"])
    else:
        train_mask = torch.ByteTensor(graph.ndata["train_mask"])
        val_mask = torch.ByteTensor(graph.ndata["val_mask"])
        test_mask = torch.ByteTensor(graph.ndata["test_mask"])
    in_feats = features.shape[1]
    # n_classes = data.num_classes
    n_edges = graph.num_edges()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        # labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # add self loop
    if args.self_loop:
        # print("self_loop")
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
    n_edges = graph.num_edges()

    if args.gpu >= 0:
        graph = graph.to(args.gpu)
    # create DGI model
    dgi = DGI(
        graph,
        in_feats,
        args.n_hidden,
        args.n_layers,
        nn.PReLU(args.n_hidden),
        args.dropout,
    )

    if cuda:
        dgi.cuda()

    dgi_optimizer = torch.optim.Adam(
        dgi.parameters(), lr=args.dgi_lr, weight_decay=args.weight_decay
    )

    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []
    for epoch in range(args.n_dgi_epochs):
        dgi.train()
        if epoch >= 3:
            t0 = time.time()

        dgi_optimizer.zero_grad()
        loss = dgi(features)
        loss.backward()
        dgi_optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(dgi.state_dict(), "best_dgi.pkl")
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print("Early stopping!")
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        """ print(
            "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
            "ETputs(KTEPS) {:.2f}".format(
                epoch, np.mean(dur), loss.item(), n_edges / np.mean(dur) / 1000
            )
        ) """

    embeds = dgi.encoder(features, corrupt=False)
    embeds = embeds.cpu().detach()

    df_node_embeddings = pd.DataFrame(data=embeds)
    df_node_embeddings

    # save
    df_node_embeddings.to_csv("{}/output/{}/node_embeddings/node-embeddings_{}_{}_{}.csv".format(dir, exp, method, group, subgroup), index=True)
    # print("Save node embeddings")

def main(exp):
    
    # %% [markdown]
    # ### Parameters

    # %%
    import json

    # dir = os.path.dirname(os.path.dirname(os.getcwd()))
    # dir = os.path.dirname(os.getcwd())
    print(dir)

    # opening JSON file
    file = open("{}/input/parameters_{}.json".format(dir, exp))
    params = json.load(file)

    exp = params["exp"]
    print("Exp:\t\t", exp)

    method = "dgi"
    print("Method:\t\t", method)

    dimension = params["dimension"]
    print("Dimension:\t", dimension)

    groups_id = params["groups_id"]
    print("Groups id:\t", groups_id)

    subgroups_id = params["subgroups_id"]
    print("Subgroups id:\t", subgroups_id)

    option = params["option"]
    print("Option:\t\t", option)

    if option:
        for group in groups_id:
            subgroups_id[group] = [option]
        print("Subgroups id:\t", subgroups_id)

    # %% [markdown]
    # ### Node embeddings

    # %%
    # custom dataset

    
    # %%
    nodes_data = pd.read_csv("{}/output/{}/preprocessing/graphs_data/nodes_data_{}_{}.csv".format(dir, exp, groups_id[0], subgroups_id[groups_id[0]][0]))
    edges_data = pd.read_csv("{}/output/{}/preprocessing/graphs_data/edges_data_{}_{}.csv".format(dir, exp, groups_id[0], subgroups_id[groups_id[0]][0]))

    dataset = CustomDataset("g1", nodes_data, edges_data)
    graph = dataset[0]

    print(graph)

    # %%
    # params

    parser = argparse.ArgumentParser(description="DGI")
    # register_data_args(parser)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        required=False,
        help="The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout probability"
    )
    parser.add_argument("--gpu", type=int, default=1, help="gpu")
    parser.add_argument(
        "--dgi-lr", type=float, default=1e-3, help="dgi learning rate"
    )
    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=1e-2,
        help="classifier learning rate",
    )
    parser.add_argument(
        "--n-dgi-epochs",
        type=int,
        default=300,
        help="number of training epochs",
    )
    parser.add_argument(
        "--n-classifier-epochs",
        type=int,
        default=300,
        help="number of training epochs",
    )
    parser.add_argument(
        "--n-hidden", type=int, default=2, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=3, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight for L2 loss"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="early stop patience condition"
    )
    parser.add_argument(
        "--self-loop",
        action="store_true",
        help="graph self-loop (default=False)",
    )
    parser.set_defaults(self_loop=True)
    parser.set_defaults(n_hidden=dimension)
    parser.set_defaults(n_layers=3)
    args = parser.parse_args("")

    print(args)

    # %%
    # get node embeddings

    for group in tqdm(groups_id):
        for subgroup in tqdm(subgroups_id[group]):
            nodes_data = pd.read_csv("{}/output/{}/preprocessing/graphs_data/nodes_data_{}_{}.csv".format(dir, exp, group, subgroup))
            edges_data = pd.read_csv("{}/output/{}/preprocessing/graphs_data/edges_data_{}_{}.csv".format(dir, exp, group, subgroup))

            # read dataset
            # data = load_data(args)
            data = CustomDataset("g_{}_{}".format(group, subgroup), nodes_data, edges_data)
            graph = data[0]

            # train
            train_dgi(exp, graph, args, method, group, subgroup)

    # %%
    df_node_embeddings = pd.read_csv("{}/output/{}/node_embeddings/node-embeddings_{}_{}_{}.csv".format(dir, exp, method, groups_id[0], 
                                                                                                        subgroups_id[groups_id[0]][0]), index_col=0)
    df_node_embeddings.head()


