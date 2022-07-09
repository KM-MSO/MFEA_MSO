from black import out
import numpy as np
import numba as nb
import torch
import torch.nn as nn
import sys
import os
MAX_INT = sys.maxsize

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
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, device):
        self.device = "cpu"
        self.model = SurrogateModel(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criteria = nn.MSELoss()
        self.learning_rate = learning_rate
    def train(self, input, output):
        input = torch.Tensor(input.flatten()).to(self.device)
        output = torch.Tensor([output]).to(self.device)
        pred = self.model(input)
        loss = self.criteria(pred, output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, input):
        with torch.no_grad():
            input = torch.Tensor(input.flatten()).to(self.device)
            return self.model(input)

    def save_pipeline(self, path):
        pass

class AbstractTask():
    def __init__(self, *args, **kwargs) -> None:
        pass
    def __eq__(self, __o: object) -> bool:
        if self.__repr__() == __o.__repr__():
            return True
        else:
            return False
    def decode(self, x):
        pass
    def __call__(self, x):
        pass

    @staticmethod
    #@nb.jit(nopython = True)
    def func(x):
        pass

#----------------------------------------------------------------------------------------------------------------------------
#a solution is an permutation start from 0 to n - 1, k is also counted from 0 but domain is counted from 1
class IDPC_EDU_FUNC(AbstractTask):    
    def __init__(self, dataset_path, file_name, use_surrogate = True):
        self.file = str(dataset_path) + '/'  + file_name
        self.datas = {}
        self.source: int
        self.target: int
        self.num_domains: int
        self.num_nodes: int
        self.num_edges: int = int(file_name[:-5].split('x')[-1])
        self.edges: np.ndarray
        self.dim: int = int(file_name[5:].split('x')[0])
        self.name = file_name.split('.')[0]
        self.use_surrogate = use_surrogate
        self.read_data()
        if self.use_surrogate:
            self.surrogate_pipeline = SurrogatePipeline(input_dim=self.num_nodes*2, 
                                                        hidden_dim=self.num_nodes, 
                                                        output_dim=1, 
                                                        learning_rate=1e-5,
                                                        device='cuda' if torch.cuda.is_available() else 'cpu')
 
    def read_data(self):
        with open(self.file, "r") as f:
            lines = f.readlines()
            #get num_nodes and num_domains from the first line
            line0 = lines[0].split()
            self.num_nodes = int(line0[0])
            self.num_domains = int(line0[1])
            count_paths = np.zeros((self.num_nodes, self.num_nodes)).astype(np.int64)
            #edges is a dictionary with key: s_t_k and value is a list with size 2 
            self.edges = nb.typed.Dict().empty(
                key_type= nb.types.unicode_type,
                value_type= nb.typeof((0, 0)),
            )
            #get source and target from the seconde line
            line1 = lines[1].split()
            self.source = int(line1[0]) - 1
            self.target = int(line1[1]) - 1
            
            #get all edges
            lines = lines[2:]
            for line in lines:
                data = [int(x) for x in line.split()]
                self.edges[f'{data[0] - 1}_{data[1] - 1}_{count_paths[data[0] - 1][data[1] - 1]}'] = tuple([data[2], data[3]])
                count_paths[data[0] - 1][data[1] - 1] += 1
            self.count_paths = count_paths

    @staticmethod
    # @nb.njit(
    #     nb.int64(
    #         nb.typeof(np.array([[1]]).astype(np.int64)),
    #         nb.int64,
    #         nb.int64,
    #         nb.int64,
    #         nb.int64,
    #         nb.typeof(nb.typed.Dict().empty(
    #             key_type= nb.types.unicode_type,
    #             value_type= nb.typeof((0, 0)),
    #         )),
    #         nb.typeof(np.array([[1]]).astype(np.int64)),
    #     )
    # )
    def func(gene,
             source,
             target,
             num_nodes,
             num_domains,
             edges,
             count_paths,
             ):
        idx = np.argsort(-gene[0])
        cost = 0
        left_domains = [False for i in range(num_domains + 1)]
        visited_vertex = [False for i in range(num_nodes)]
    
        curr = source
        # path = []
        while(curr != target):
            visited_vertex[curr] = True
            stop = True
            for t in idx:
                if visited_vertex[t]:
                    continue
                #if there is no path between curr and t
                if count_paths[curr][t] == 0:
                    continue
                
                k = gene[1][curr] % count_paths[curr][t] 
                # key = get_key(curr, t, k)
                key = '_'.join([str(curr), str(t), str(k)])
                d = edges[key][1]
                if left_domains[d]:
                    continue
                cost += edges[key][0]
                # path.append(key +  '_' + str(edges[key][0]) + '_' + str(d))
                left_domains[d] = True
                curr = t
                stop = False
                break
            if stop:
                return MAX_INT
        return cost
        
    def __call__(self, gene: np.ndarray):
        # decode
        # idx = np.argsort(gene[0])[:self.dim]
        idx = np.arange(self.dim)
        gene = np.ascontiguousarray(gene[:, idx])
        # eval
        if self.use_surrogate:
            if os.environ.get("surrogate_status") == "train":
                cost = __class__.func(gene, self.source, self.target,
                                self.num_nodes, self.num_domains, self.edges, self.count_paths)
                self.surrogate_pipeline.train(gene, cost)
            else:
                cost = self.surrogate_pipeline.predict(gene).item()
        else:
            cost = __class__.func(gene, self.source, self.target,
                                self.num_nodes, self.num_domains, self.edges, self.count_paths)
        return cost
