import torch.nn as nn
import torch 
import torch.nn.functional as F
import torch_geometric.nn as gnn_nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

import numpy as np
#import ray

class GraphDataset(InMemoryDataset):
    def __init__(self, tasks, root = './data', transform= None, pre_transform= None, pre_filter= None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = []
        self.tasks= tasks
       
    @staticmethod 
    # @ray.remote
    def create_new_graph_instance(edge_index, edge_attribute, num_nodes: int, source:int, target:int, genes: np.ndarray, y: int):
        x = [[0, genes[0, i], genes[1, i]] for i in range(num_nodes)]
        x[source][0] = -1
        x[target][0] = 1
            
        x = torch.tensor(x, dtype= torch.float)
        
        
        
        return Data(x= x, edge_index= edge_index, edge_attr= edge_attribute, y= torch.Tensor(y))
    
    def append(self, genes, costs, skfs):
        for i in range(genes.shape[0]):
            self.data.append(__class__.create_new_graph_instance(
                edge_index = self.tasks[skfs[i]].edge_index,
                edge_attribute= self.tasks[skfs[i]].edge_attribute,
                num_nodes= self.tasks[skfs[i]].num_nodes, 
                source= self.tasks[skfs[i]].source,
                target= self.tasks[skfs[i]].target,
                genes = genes[i], 
                y = costs[i],
            ))
        # new_data =  [
        #     __class__.create_new_graph_instance.remote(
        #         self.tasks[skfs[i]].edge_index,
        #         self.tasks[skfs[i]].edge_attribute,
        #         self.tasks[skfs[i]].num_nodes, 
        #         self.tasks[skfs[i]].source,
        #         self.tasks[skfs[i]].target,
        #         genes[i], 
        #         costs[i],
        #     ) for i in range(genes.shape[0])
        # ]
        # self.data.extend(ray.get(new_data))
            
    def len(self):
        return len(self.data)
            
    def get(self, idx):
        return self.data[idx]
        
    def _download(self):
        pass
    
    def _process(self):
        pass
    
    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__)
  
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
    def __init__(self, in_channels, hid_channels, num_nodes):
        super().__init__()
        self.gc1 = gnn_nn.GATConv(in_channels=in_channels, out_channels=hid_channels)
        self.gc2 = gnn_nn.GATConv(in_channels=hid_channels, out_channels=1)
        self.fc = nn.Linear(num_nodes, 1)
        
    def forward(self, inputs):
        vertices_feature, edge_index, edge_attr = inputs.x, inputs.edge_index, inputs.edge_attr
        x = F.relu(self.gc1(vertices_feature, edge_index, edge_attr))
        x = self.gc2(vertices_feature, edge_index, edge_attr)
        x = x.squeeze(1)
        x = self.fc(x)
        return x
    
class SurrogatePipeline():
    def __init__(self, input_dim, hidden_dim, num_nodes, learning_rate, epochs = 100, use_cuda = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu') 
        self.model = SurrogateModel(input_dim, hidden_dim, num_nodes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criteria = nn.MSELoss()
        self.epochs = epochs

        
    def train(self, datasets):
        print("Training surrogate")
        self.set_train()

        self.model.train()
        datasets.to(self.device)

        dataloader = DataLoader(datasets, batch_size=2, shuffle=True)

        for epoch in range(self.epochs):
            losses = []
            for batch_idx, (inputs, outputs) in enumerate(dataloader):
                preds = self.model(inputs)
                loss = self.criteria(preds, outputs)
                losses.append(loss.item())
                
                print(f'Epoch {epoch} - Batch {batch_idx} - Loss: {loss.item()}')
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch} - Loss: {np.mean(losses)}')

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