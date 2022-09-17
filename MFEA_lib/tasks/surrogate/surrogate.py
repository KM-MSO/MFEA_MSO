import torch.nn as nn
import torch 
import torch_geometric.nn as gnn
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import ray

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
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, device = 'cpu'):
        self.device = device 
        self.model = SurrogateModel(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criteria = nn.MSELoss()
        self.learning_rate = learning_rate
        
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