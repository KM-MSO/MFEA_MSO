from asyncio import threads
from http.server import ThreadingHTTPServer
from re import T
import torch.nn as nn
import torch 
import torch.nn.functional as F
import torch_geometric.nn as gnn_nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import ray
import sys
MAX_INT = sys.maxsize

class stacking_data(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        return None


class GraphDataset(InMemoryDataset):
    def __init__(self, tasks, root = './data', transform= None, pre_transform= None, pre_filter= None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = []
        self.tasks= tasks
        self.latest_data = []
       
    @staticmethod 
    # @ray.remote
    def create_new_graph_instance(edge_index, edge_attribute, num_nodes: int, source:int, target:int, genes: np.ndarray, y: int):
        x = [[0, genes[0, i], genes[1, i]] for i in range(num_nodes)]
        x[source][0] = -1
        x[target][0] = 1
            
        x = torch.tensor(x, dtype= torch.float)
        
        return Data(x= x, edge_index= edge_index, edge_attr= edge_attribute, y= torch.Tensor([y]))
    
    def append(self, genes, costs, skfs):
        self.latest_data = []
        for i in range(genes.shape[0]):
            self.latest_data.append(__class__.create_new_graph_instance(
                edge_index = self.tasks[skfs[i]].edge_index,
                edge_attribute= self.tasks[skfs[i]].edge_attribute,
                num_nodes= self.tasks[skfs[i]].num_nodes, 
                source= self.tasks[skfs[i]].source,
                target= self.tasks[skfs[i]].target,
                genes = genes[i], 
                y = costs[i],
            ))
        self.data.extend(self.latest_data)

    def len(self):
        return len(self.data)
            
    def get(self, idx):
        return self.data[idx]
        
    def _download(self):
        pass
    

         
    def _process(self):
        pass
    
    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
  
# type 2
class GNN_GCN(nn.Module):
    def __init__(self, in_channels, hid_channels, num_nodes):
        super(GNN_GCN, self).__init__()
        self.gc1 = gnn_nn.GCNConv(in_channels, hid_channels)
        self.gc2 = gnn_nn.GCNConv(hid_channels, 1)
        self.fc = nn.Linear(num_nodes, 1)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.gc1(x, edge_index, edge_weight))
        x = self.gc2(x, edge_index, edge_weight)
        x = x.squeeze(1)
        x = self.fc(x)
        return x

# type 1
class SurrogateModel(nn.Module):
    def __init__(self, in_channels, hid_channels, threshhold = 0.5):
        super().__init__()
        self.gc1 = gnn_nn.GATConv(in_channels=in_channels, out_channels=hid_channels)
        self.gc2 = gnn_nn.GATConv(in_channels=hid_channels, out_channels=32)
        self.lstm = nn.LSTM(32, 32)
        self.fc1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32,1)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
            )
        
        
    def forward(self, inputs):
        vertices_feature, edge_index, edge_attr = inputs.x, inputs.edge_index, inputs.edge_attr
        features = F.relu(self.gc1(vertices_feature, edge_index, edge_attr))
        features = self.gc2(features, edge_index, edge_attr)

        # classification head
        x_c = torch.mean(features, dim = 0)
        x_c = self.fc2(x_c)

        # regression head
        x_r = features.squeeze(1)
        _, (_, x_r) = self.lstm(x_r)
        x_r = self.fc1(x_r)
        x_r = x_r.flatten()

        x = torch.cat((x_c.unsqueeze(0), x_r.unsqueeze(0)),dim=0)
        return x
    
class SurrogatePipeline():
    def __init__(self, input_dim, hidden_dim, learning_rate, epochs = 1, threshold = 0.5, use_cuda = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu') 
        self.model = SurrogateModel(input_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criteria_regress = nn.MSELoss()
        self.criteria_classify = nn.BCELoss()
        self.epochs = epochs
        self.threshold = threshold

        
    def train(self, datasets):
        print("\n Training surrogate")

        self.model.train()

        dataloader = DataLoader(datasets, batch_size=1, shuffle=True, pin_memory= True, num_workers= 2)

        for epoch in range(self.epochs):
            losses = {'loss': [], 'loss_regress': [], 'loss_classify': []}
            
            for _, batch in tqdm(enumerate(dataloader)):
                preds = self.model(batch.to(self.device))
                logits = (preds[0] > self.threshold).float()
                gt_logits = (batch.y < MAX_INT).float()

                loss_regress = self.criteria_regress(preds[1], batch.y) 
                loss_classify =  self.criteria_classify(preds[0], gt_logits)

                alpha = (logits + gt_logits < 2).float()

                loss = 1e-6*(1-alpha)*loss_regress + loss_classify
                losses['loss'].append(loss.item())
                losses['loss_classify'].append(loss_classify.item())
                losses['loss_regress'].append(loss_regress.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch} | Loss: {np.mean(losses['loss'])} | Loss_clasify: {np.mean(losses['loss_classify'])} | Loss_regress: {np.mean(losses['loss_regress'])}")

    def eval(self, inputs):
        pass

    def predict(self, input):
        with torch.no_grad():
            input.to(self.device)
            pred = self.model(input)
            return pred

    def save_model(self):
        pass

    def load_model(self):
        pass

    def save_pipeline(self, path):
        pass