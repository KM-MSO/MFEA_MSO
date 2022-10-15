from torch_geometric.data import Data, InMemoryDataset
import torch
import numpy as np
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
    def create_new_graph_instance(edge_index, edge_attribute, num_nodes: int, source:int, target:int, genes: np.ndarray, y: int, thresh_hold: int = None):
        x = [[0, genes[0, i], genes[1, i]] for i in range(num_nodes)]
        x[source][0] = -1
        x[target][0] = 1
            
        x = torch.tensor(x, dtype= torch.float)
        return Data(x= x, edge_index= edge_index, edge_attr= edge_attribute, y= torch.Tensor([y]),
                    thresh_hold = torch.Tensor([1 if y < thresh_hold else 0]))
    
    def append(self, genes, costs, skfs, thresh_hold):
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
                thresh_hold= thresh_hold[i],
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
        return '{}{}'.format(self.__class__.__name__)