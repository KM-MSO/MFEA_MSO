import torch.nn as nn
import torch 
class SurrogateModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
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