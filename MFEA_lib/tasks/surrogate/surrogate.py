import torch.nn as nn
import torch 
import torch_geometric.nn as gnn
from torch_geometric.data import Data, InMemoryDataset
import numpy as np

class GraphDataset(InMemoryDataset):
    def __init__(self, edges: dict, count_paths: np.ndarray, num_nodes: int, source:int, target:int, transform= None):
        x = [[0] for _ in range(num_nodes)]
        x[source] = [-1]
        x[target] = [1]
        x = torch.tensor(x, dtype= torch.float)
        
        edge_index = []
        edge_attribute = []
        for i in range(count_paths.shape[0]):
            for j in range(count_paths.shape[1]):
                n = count_paths[i][j]

                edge_index.extend( [[i , j] for _ in range(n)] )
                edge_attribute.extend( [edges.get(f'{i}_{j}_{k}') for k in range(n)] )
                
        edge_index = torch.tensor(edge_index, dtype= torch.long).reshape(2, -1)
        edge_attribute = torch.tensor(edge_attribute, dtype= torch.long)
        
        self.graph = Data(x= x, edge_index= edge_index, edge_attr= edge_attribute)
        
    def _download(self):
        pass
    
    def _process(self):
        pass
    
    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__)
  

class GNNModel(nn.Module):
    def __init__(self):
        pass

class SurrogateModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.edge_conv1 = gnn.EdgeConv(nn.Linear(2, 64))
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, graph):
        graph_features = self.edge_conv1(graph.x, graph.data)
        print(x.shape, graph_features.shape)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
class SurrogatePipeline():
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, graph, device= None):
        self.device = device if device else 'cpu'
        self.model = SurrogateModel(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criteria = nn.MSELoss()
        self.learning_rate = learning_rate
        self.graph = graph
        
    def train(self, input, output):
        input = torch.Tensor(input.flatten()).to(self.device)
        output = torch.Tensor([output]).to(self.device)
        pred = self.model(input)
        loss = self.criteria(pred, output)
        print(f'Loss: {loss}')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, input):
        with torch.no_grad():
            input = torch.Tensor(input.flatten()).to(self.device)
            return self.model(input)

    def save_pipeline(self, path):
        pass