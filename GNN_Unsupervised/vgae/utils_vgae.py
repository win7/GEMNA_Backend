import argparse
import os
import time

import dgl

from vgae import model
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
# from input_data import load_data

import torch
from dgl.data import DGLDataset

from vgae.preprocess import (
    mask_test_edges,
    mask_test_edges_dgl,
    preprocess_graph,
    sparse_to_tuple,
)
from sklearn.metrics import average_precision_score, roc_auc_score

from tqdm import tqdm
import pandas as pd
import networkx as nx
import numpy as np

# os.environ["DGLBACKEND"] = "pytorch"
    
def args_vgae(dimension):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    parser = argparse.ArgumentParser(description="Variant Graph Auto Encoder")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Initial learning rate."
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=300, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden1",
        "-h1",
        type=int,
        default=32,
        help="Number of units in hidden layer 1.",
    )
    parser.add_argument(
        "--hidden2",
        "-h2",
        type=int,
        default=dimension,
        help="Number of units in hidden layer 2.",
    )
    parser.add_argument(
        "--datasrc",
        "-s",
        type=str,
        default="dgl",
        help="Dataset download from dgl Dataset or website.",
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="cora", help="Dataset string."
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")
    args = parser.parse_args("")
    
    # roc_means = []
    # ap_means = []
    return args

class CustomDatasetVGAE(DGLDataset):
    def __init__(self, name, nodes_data, edges_data):
        self.dir = dir
        self.nodes_data = nodes_data
        self.edges_data = edges_data
        super().__init__(name=name)
       
    def process(self):
        node_features = torch.from_numpy(self.nodes_data.to_numpy())
        # node_features = torch.from_numpy(np.log10(self.nodes_data["degree"].to_numpy()))
        node_features = node_features.to(torch.float32)
        # node_features = torch.reshape(node_features, (-1, 1))

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
    
def compute_loss_para(adj, device):
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def train_vgae(exp, graph, args, method, group, subgroup, iteration):
     # check device
    device = torch.device(
        "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
    )
    print("-------------------")
    print(device)
    print("-------------------")
    
    # device = "cpu"
    # device
    
    # torch.cuda.set_device(device)
    torch.cuda.set_device(0)
    # extract node features
    # print(device)
    feats = graph.ndata.pop("feat").cuda() # to(device)
    in_dim = feats.shape[-1]
    # print(in_dim)

    # generate input
    adj_orig = graph.adj_external().to_dense()

    # build test set with 10% positive links
    (
        train_edge_idx,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    ) = mask_test_edges_dgl(graph, adj_orig)

    graph = graph.to(device)

    # create train graph
    train_edge_idx = torch.tensor(train_edge_idx).to(device)
    train_graph = dgl.edge_subgraph(graph, train_edge_idx, relabel_nodes=False)
    train_graph = train_graph.to(device)
    adj = train_graph.adj_external().to_dense().to(device)

    # compute loss parameters
    weight_tensor, norm = compute_loss_para(adj, device)

    # create model
    vgae_model = model.VGAEModel(in_dim, args.hidden1, args.hidden2)
    vgae_model = vgae_model.to(device)

    # create training component
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
    """ print(
        "Total Parameters:",
        sum([p.nelement() for p in vgae_model.parameters()]),
    ) """

    # create training epoch
    for epoch in tqdm(range(args.epochs)):
        t = time.time()

        # Training and validation using a full graph
        vgae_model.train()

        logits = vgae_model.forward(graph, feats)

        # compute loss
        loss = norm * F.binary_cross_entropy(
            logits.view(-1), adj.view(-1), weight=weight_tensor
        )
        kl_divergence = (
            0.5
            / logits.size(0)
            * (
                1
                + 2 * vgae_model.log_std
                - vgae_model.mean**2
                - torch.exp(vgae_model.log_std) ** 2
            )
            .sum(1)
            .mean()
        )
        loss -= kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train_acc = get_acc(logits, adj)

        # val_roc, val_ap = get_scores(val_edges, val_edges_false, logits)

        # Print out performance
        """ print(
            "Epoch:",
            "%04d" % (epoch + 1),
            "train_loss=",
            "{:.5f}".format(loss.item()),
            "train_acc=",
            "{:.5f}".format(train_acc),
            "val_roc=",
            "{:.5f}".format(val_roc),
            "val_ap=",
            "{:.5f}".format(val_ap),
            "time=",
            "{:.5f}".format(time.time() - t),
        ) """

    """ test_roc, test_ap = get_scores(test_edges, test_edges_false, logits)
    # roc_means.append(test_roc)
    # ap_means.append(test_ap)
    print(
        "End of training!",
        "test_roc=",
        "{:.5f}".format(test_roc),
        "test_ap=",
        "{:.5f}".format(test_ap),
    ) """

    embeds = vgae_model.encoder(graph, feats)
    embeds = embeds.cpu().detach()

    df_node_embeddings = pd.DataFrame(data=embeds)
    df_node_embeddings

    # save
    df_node_embeddings.to_csv("output/{}/node_embeddings/node-embeddings_{}_{}_{}_{}.csv".format(exp, method, group, subgroup, iteration), index=True)
    # print("Save node embeddings")