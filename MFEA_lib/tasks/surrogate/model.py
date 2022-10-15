import torch.nn as nn
import torch 
import torch.nn.functional as F
import torch_geometric.nn as gnn_nn
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

  
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
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.gc1 = gnn_nn.GATConv(in_channels=in_channels, out_channels=hid_channels)
        self.gc2 = gnn_nn.GATConv(in_channels=hid_channels, out_channels=256)
        self.lstm = nn.LSTM(256, 256)
        self.regress = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256,1)
            )
        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )
        
    def forward(self, inputs):
        vertices_feature, edge_index, edge_attr = inputs.x, inputs.edge_index, inputs.edge_attr
        x = F.relu(self.gc1(vertices_feature, edge_index, edge_attr))
        x = self.gc2(vertices_feature, edge_index, edge_attr)
        x = x.squeeze(1)
        _, (_, x) = self.lstm(x)
        v = self.regress(x)
        c = self.classify(x)
        return v.flatten(), c.flatten()
    
class SurrogatePipeline():
    def __init__(self, input_dim, hidden_dim, learning_rate, epochs = 1, device = 'cpu'):
        self.device = device
        self.model = SurrogateModel(input_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reg_criteria = nn.MSELoss()
        self.cls_criteria = nn.BCELoss()
        self.epochs = epochs

        
    def train(self, datasets):
        print("Training surrogate")

        self.model.train()

        dataloader = DataLoader(datasets, batch_size=1, shuffle=True, pin_memory= True, num_workers= 2)

        for epoch in range(self.epochs):
            losses = []
            
            for _, batch in tqdm(enumerate(dataloader)):
                vpreds, cpreds = self.model(batch.cuda())
                loss = self.reg_criteria(vpreds / batch.y, torch.Tensor([1]).type(torch.float).cuda()) + self.cls_criteria(cpreds, batch.thresh_hold)
                losses.append(loss.item())
                
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