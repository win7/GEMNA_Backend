import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GINConv, VGAE
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.utils import to_undirected

# custom dataset
class CustomDataset(InMemoryDataset):
    def __init__(self, nodes_data, edges_data, transform=None):
        self.nodes_data = nodes_data
        self.edges_data = edges_data
        super().__init__('.', transform, None, None)
        
        x_ = self.nodes_data.values
        x = torch.tensor(x_, dtype=torch.float)

        edge_index_ = self.edges_data.iloc[:, [0, 1]].values
        edge_index = torch.tensor(edge_index_, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, x.shape[0])

        data = Data(x=x, edge_index=edge_index)
        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    
    def info(self):
        print("Validate:\t {}".format(self.data.validate(raise_on_error=True)))
        print("Num. nodes:\t {}".format(self.data.num_nodes))
        print("Num. edges:\t {}".format(self.data.num_edges))
        print("Num. features:\t {}".format(self.data.num_node_features))
        print("Has isolated:\t {}".format(self.data.has_isolated_nodes()))
        print("Has loops:\t {}".format(self.data.has_self_loops()))
        print("Is directed:\t {}".format(self.data.is_directed()))
        print("Is undirected:\t {}".format(self.data.is_undirected()))

class Encoder(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv1 = GCNConv(dim_in, 2 * dim_out)
        self.conv_mu = GCNConv(2 * dim_out, dim_out)
        self.conv_logstd = GCNConv(2 * dim_out, dim_out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class Encoder_(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_h = 2 * dim_out
        self.conv1 = GINConv(
            Sequential(Linear(dim_in, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        """ self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                    Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                      Linear(dim_h, dim_h), ReLU())) """
        """ self.conv_mu = GINConv(
            Sequential(Linear(2 * dim_out, 2 * dim_out),
                       BatchNorm1d(2 * dim_out), ReLU(),
                       Linear(2 * dim_out, dim_out), ReLU()))
        self.conv_logstd = GINConv(
            Sequential(Linear(2 * dim_out, 2 * dim_out),
                       BatchNorm1d(2 * dim_out), ReLU(),
                       Linear(2 * dim_out, dim_out))) """
        # self.conv1 = GCNConv(dim_in, 2 * dim_out)
        self.conv_mu = GCNConv(2 * dim_out, dim_out)
        self.conv_logstd = GCNConv(2 * dim_out, dim_out)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        
def train(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index) + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

def train_vgae_tg(exp, model, train_data, test_data, method, group, subgroup, iteration):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
    for epoch in range(301):
        loss = train(model, optimizer, train_data)
        val_auc, val_ap = test(model, test_data)
        """ if epoch % 50 == 0:
            print(f'Epoch {epoch:>2} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}') """

    # test_auc, test_ap = test(model, test_data)
    # print(f'Test AUC: {test_auc:.4f} | Test AP {test_ap:.4f}')
    
    model.eval()
    z = model.encode(train_data.x, train_data.edge_index)
    z = z.detach().cpu()
    
    df_node_embeddings = pd.DataFrame(data=z)
    df_node_embeddings

    # save
    df_node_embeddings.to_csv("output/{}/node_embeddings/node-embeddings_{}_{}_{}_{}.csv".format(exp, method, group, subgroup, iteration), index=True)
    # print("Save node embeddings")