from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import (
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import (
    VGAE,
    DeepGraphInfomax,
    ARGVA,
    SAGEConv,
    GCNConv,
    GINConv,
    GraphConv
)
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

import pandas as pd
import torch
import torch_geometric.transforms as T

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

class VGAE_Base(object):
    def __init__(self, dataset, dimension, device):
        self.dataset = dataset
        self.train_data, self.val_data, self.test_data = self.dataset[0]
        # self.train_data = self.train_data.to(device, 'x', 'edge_index')
        
        self.in_channels = self.dataset.num_features
        self.out_channels = dimension
        
        self.device = device
        self.model = VGAE(self.VariationalGCNEncoder(self.in_channels, self.out_channels)).to(self.device)
        self.model = torch.compile(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
    
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, data):
        self.model.eval()
        z = self.model.encode(data.x, data.edge_index)
        return self.model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    
    def fit(self, epochs):
        loop_obj = tqdm(range(epochs))
        for epoch in loop_obj:
            loop_obj.set_description(f"Epoch: {epoch + 1}")
            
            loss = self.train()
            loop_obj.set_postfix_str(f"Loss: {loss:.4f}")

            # val_auc, val_ap = self.test(self.test_data)
            """ if epoch % 50 == 0:
                print(f'Epoch {epoch:>2} | Loss: {loss:.4f}') # | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}') """

        # test_auc, test_ap = self.test(self.test_data)
        # print(f'Test AUC: {test_auc:.4f} | Test AP {test_ap:.4f}')

    def get_node_embeddings(self):
        self.model.eval()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        z = z.detach().cpu().numpy()
        return z
    
    def save_node_embeddings(self, path):
        z = self.get_node_embeddings()
        
        df_node_embeddings = pd.DataFrame(data=z)
        df_node_embeddings.to_csv(path, index=True)

    class VariationalGCNEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 2 * out_channels)
            self.conv_mu = GCNConv(2 * out_channels, out_channels)
            self.conv_logstd = GCNConv(2 * out_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class VGAE_Base_(object):   
    def __init__(self, dataset, dimension, device):
        self.dataset = dataset
        self.train_data, self.val_data, self.test_data = self.dataset[0]
        # self.train_data = self.train_data.to(device, 'x', 'edge_index')
        
        self.in_channels = self.dataset.num_features
        self.out_channels = dimension
        
        self.device = device
        self.model = VGAE(self.VariationalGCNEncoder(self.in_channels, self.out_channels)).to(self.device)
        # self.model = torch.compile(self.model, dynamic=False, fullgraph=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
    
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, data):
        self.model.eval()
        z = self.model.encode(data.x, data.edge_index)
        return self.model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    
    def fit(self, epochs):
        for epoch in range(epochs):
            loss = self.train()
            if epoch % 10 == 0:
                print(loss)
            # val_auc, val_ap = self.test(self.test_data)
            """ if epoch % 50 == 0:
                print(f'Epoch {epoch:>2} | Loss: {loss:.4f}') # | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}') """

        # test_auc, test_ap = self.test(self.test_data)
        # print(f'Test AUC: {test_auc:.4f} | Test AP {test_ap:.4f}')

    def get_node_embeddings(self):
        self.model.eval()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        z = z.detach().cpu().numpy()
        return z
    
    def save_node_embeddings(self, path):
        z = self.get_node_embeddings()
        
        df_node_embeddings = pd.DataFrame(data=z)
        df_node_embeddings.to_csv(path, index=True)

    class VariationalGCNEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = GraphConv(in_channels, 2 * out_channels)
            self.conv_mu = GraphConv(2 * out_channels, out_channels)
            self.conv_logstd = GraphConv(2 * out_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class DGI_Transductive(object):
    def __init__(self, dataset, dimension, device):
        self.dataset = dataset
        self.data = self.dataset[0]
        
        self.in_channels = self.dataset.num_features
        self.out_channels = dimension
        
        self.device = device
        self.model = DeepGraphInfomax(
            hidden_channels=dimension,
            encoder=self.Encoder(self.in_channels, self.out_channels),
            summary=lambda z, *args, **kwargs: z.mean(dim=0).sigmoid(),
            corruption=self.corruption,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
    
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        pos_z, neg_z, summary = self.model(self.data.x, self.data.edge_index)
        loss = self.model.loss(pos_z, neg_z, summary)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(self):
        self.model.eval()
        z, _, _ = self.model(self.data.x, self.data.edge_index)
        acc = self.model.test(z[self.data.train_mask], self.data.y[self.data.train_mask],
                        z[self.data.test_mask], self.data.y[self.data.test_mask], max_iter=150)
        return acc
    
    def fit(self, epochs):
        for epoch in range(epochs):
            loss = self.train()
            """ if epoch % 50 == 0:
                print(f'Epoch {epoch:>2} | Loss: {loss:.4f}') """
        # acc = test()
        # print(f'Accuracy: {acc:.4f}')

    def get_node_embeddings(self):
        self.model.eval()
        z, _, _ = self.model(self.data.x, self.data.edge_index)
        z = z.detach().cpu().numpy()
        return z
    
    def save_node_embeddings(self, path):
        z = self.get_node_embeddings()
        
        df_node_embeddings = pd.DataFrame(data=z)
        df_node_embeddings.to_csv(path, index=True)
    
    def corruption(self, x, edge_index):
        return x[torch.randperm(x.size(0), device=x.device)], edge_index
    
    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super().__init__()
            self.conv = GCNConv(in_channels, hidden_channels)
            self.prelu = torch.nn.PReLU(hidden_channels)

        def forward(self, x, edge_index):
            x = self.conv(x, edge_index)
            x = self.prelu(x)
            return x
        
class DGI_Inductive(object):   
    def __init__(self, dataset, dimension, device):
        self.dataset = dataset
        self.data = self.dataset[0].to(device, 'x', 'edge_index')
        
        self.train_loader = NeighborLoader(self.data, num_neighbors=[10, 10, 25], batch_size=256, shuffle=True, num_workers=12)
        self.test_loader = NeighborLoader(self.data, num_neighbors=[10, 10, 25], batch_size=256, num_workers=12)

        self.in_channels = self.dataset.num_features
        self.out_channels = dimension
        
        self.device = device
        self.model = DeepGraphInfomax(
            hidden_channels=dimension, encoder=self.Encoder(self.in_channels, self.out_channels),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=self.corruption).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train(self):
        self.model.train()
        total_loss = total_examples = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            pos_z, neg_z, summary = self.model(batch.x, batch.edge_index, batch.batch_size)
            loss = self.model.loss(pos_z, neg_z, summary)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * pos_z.size(0)
            total_examples += pos_z.size(0)
        return total_loss / total_examples

    @torch.no_grad()
    def test(self):
        self.model.eval()
        zs = []
        for batch in self.test_loader:
            pos_z, _, _ = self.model(batch.x, batch.edge_index, batch.batch_size)
            zs.append(pos_z.cpu())
        z = torch.cat(zs, dim=0)
        train_val_mask = self.data.train_mask | self.data.val_mask
        acc = self.model.test(z[train_val_mask], self.data.y[train_val_mask],
                        z[self.data.test_mask], self.data.y[self.data.test_mask], max_iter=10000)
        return acc
    
    def fit(self, epochs):
        for epoch in range(epochs):
            loss = self.train()
            """ if epoch % 50 == 0:
                print(f'Epoch {epoch:>2} | Loss: {loss:.4f}') """
        # test_acc = test()
        # print(f'Test Accuracy: {test_acc:.4f}')
        
    def get_node_embeddings(self):
        self.model.eval()
        zs = []
        for batch in self.test_loader:
            pos_z, _, _ = self.model(batch.x, batch.edge_index, batch.batch_size)
            zs.append(pos_z.cpu())
        z = torch.cat(zs, dim=0)
        z = z.detach().cpu().numpy()
        return z
    
    def save_node_embeddings(self, path):
        z = self.get_node_embeddings()
        
        df_node_embeddings = pd.DataFrame(data=z)
        df_node_embeddings.to_csv(path, index=True)

    def corruption(self, x, edge_index, batch_size):
        return x[torch.randperm(x.size(0))], edge_index, batch_size
    
    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super().__init__()
            self.convs = torch.nn.ModuleList([
                SAGEConv(in_channels, hidden_channels),
                SAGEConv(hidden_channels, hidden_channels),
                SAGEConv(hidden_channels, hidden_channels)
            ])

            self.activations = torch.nn.ModuleList()
            self.activations.extend([
                torch.nn.PReLU(hidden_channels),
                torch.nn.PReLU(hidden_channels),
                torch.nn.PReLU(hidden_channels)
            ])

        def forward(self, x, edge_index, batch_size):
            for conv, act in zip(self.convs, self.activations):
                x = conv(x, edge_index)
                x = act(x)
            return x[:batch_size]
    
class ARGVA_Base(object):   
    def __init__(self, dataset, dimension, device):
        self.dataset = dataset
        self.train_data, self.val_data, self.test_data = self.dataset[0]
        
        self.in_channels = self.train_data.num_features
        self.out_channels = dimension
        self.hidden_channels = 32
        
        self.device = device
        self.encoder = self.Encoder(self.in_channels, hidden_channels=self.hidden_channels, out_channels=self.out_channels)
        self.discriminator = self.Discriminator(in_channels=self.out_channels, hidden_channels=self.hidden_channels * 2, out_channels=self.out_channels)
        self.model = ARGVA(self.encoder, self.discriminator).to(self.device)

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.005)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
    
    def train(self):
        self.model.train()
        self.encoder_optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)

        # We optimize the discriminator more frequently than the encoder.
        for i in range(5):
            self.discriminator_optimizer.zero_grad()
            discriminator_loss = self.model.discriminator_loss(z)
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + self.model.reg_loss(z)
        loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.encoder_optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, data):
        self.model.eval()
        z = self.model.encode(data.x, data.edge_index)

        # Cluster embedded values using k-means.
        kmeans_input = z.cpu().numpy()
        kmeans = KMeans(n_clusters=7, random_state=0).fit(kmeans_input)
        pred = kmeans.predict(kmeans_input)

        labels = data.y.cpu().numpy()
        completeness = completeness_score(labels, pred)
        hm = homogeneity_score(labels, pred)
        nmi = v_measure_score(labels, pred)

        auc, ap = self.model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
        return auc, ap, completeness, hm, nmi
    
    def fit(self, epochs):
        for epoch in range(epochs):
            loss = self.train()
            # auc, ap, completeness, hm, nmi = test(test_data)
            """ if epoch % 50 == 0:
                print(f'Epoch {epoch:>2} | Loss: {loss:.4f}') """

    def get_node_embeddings(self):
        self.model.eval()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        z = z.detach().cpu().numpy()
        return z
    
    def save_node_embeddings(self, path):
        z = self.get_node_embeddings()
        
        df_node_embeddings = pd.DataFrame(data=z)
        df_node_embeddings.to_csv(path, index=True)

    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv_mu = GCNConv(hidden_channels, out_channels)
            self.conv_logstd = GCNConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    class Discriminator(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.lin1 = Linear(in_channels, hidden_channels)
            self.lin2 = Linear(hidden_channels, hidden_channels)
            self.lin3 = Linear(hidden_channels, out_channels)

        def forward(self, x):
            x = self.lin1(x).relu()
            x = self.lin2(x).relu()
            return self.lin3(x)

class VGAE_Linear(object):   
    def __init__(self, dataset, dimension, device):
        self.dataset = dataset
        self.train_data, self.val_data, self.test_data = self.dataset[0]
        
        self.in_channels = self.dataset.num_features
        self.out_channels = dimension
        
        self.device = device
        self.model = VGAE(self.VariationalLinearEncoder(self.in_channels, self.out_channels)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, data):
        self.model.eval()
        z = self.model.encode(data.x, data.edge_index)
        return self.model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    
    def fit(self, epochs):
        for epoch in range(epochs):
            loss = self.train()
            # val_auc, val_ap = test(test_data)
            """ if epoch % 50 == 0:
                print(f'Epoch {epoch:>2} | Loss: {loss:.4f}') # | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}') """

        # test_auc, test_ap = test(test_data)
        # print(f'Test AUC: {test_auc:.4f} | Test AP {test_ap:.4f}')

    def get_node_embeddings(self):
        self.model.eval()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        z = z.detach().cpu().numpy()
        return z
    
    def save_node_embeddings(self, path):
        z = self.get_node_embeddings()
        
        df_node_embeddings = pd.DataFrame(data=z)
        df_node_embeddings.to_csv(path, index=True)

    class VariationalLinearEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_mu = GCNConv(in_channels, out_channels)
            self.conv_logstd = GCNConv(in_channels, out_channels)

        def forward(self, x, edge_index):
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
