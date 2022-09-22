import numpy as np
import numba as nb
import sys
MAX_INT = sys.maxsize
import ray
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
    @nb.jit(nopython = True)
    def func(x):
        pass

    

#@ray.remote
def create_idpc(dir, file):
    with open(str(dir) + '/'  + file, "r") as f:
        lines = f.readlines()
        #get num_nodes and num_domains from the first line
        line0 = lines[0].split()
        num_nodes = int(line0[0])
        num_domains = int(line0[1])
        count_paths = np.zeros((num_nodes, num_nodes)).astype(np.int64)
        #edges is a dictionary with key: s_t_k and value is a list with size 2 
        edges = {}
        # normal_edges = {}
        #get source and target from the seconde line
        line1 = lines[1].split()
        source = int(line1[0]) - 1
        target = int(line1[1]) - 1
        
        #get all edges
        lines = lines[2:]
        for line in lines:
            data = [int(x) for x in line.split()]
            edges[f'{data[0] - 1}_{data[1] - 1}_{count_paths[data[0] - 1][data[1] - 1]}'] = tuple([data[2], data[3]])
            count_paths[data[0] - 1][data[1] - 1] += 1
    return IDPC_EDU_FUNC(dir, file, source, target, num_domains, num_nodes, count_paths, edges)

#----------------------------------------------------------------------------------------------------------------------------
#a solution is an permutation start from 0 to n - 1, k is also counted from 0 but domain is counted from 1
class IDPC_EDU_FUNC(AbstractTask):        
    def __init__(self, dataset_path, file_name, source, target, num_domains, num_nodes, count_paths, edges):
        self.file = str(dataset_path) + '/'  + file_name
        self.datas = {}
        self.source: int
        self.target: int
        self.num_domains: int
        self.num_nodes: int
        self.num_edges: int = int(file_name[:-5].split('x')[-1])
        self.dim: int = int(file_name[5:].split('x')[0])
        self.name = file_name.split('.')[0]
        self.source, self.target, self.num_domains, self.num_nodes, self.count_paths, self.edges = source, target, num_domains, num_nodes, count_paths, edges
        self.edge_index = []
        self.edge_attribute = []
        self.edge_weight = []
        self.max_fitness = MAX_INT
        # type 1
        for i in range(self.count_paths.shape[0]):
            for j in range(self.count_paths.shape[1]):
                n = self.count_paths[i][j]

                self.edge_index.extend( [[i , j] for _ in range(n)] )
                self.edge_attribute.extend( [self.edges.get(f'{i}_{j}_{k}') for k in range(n)] )

                
        # type 2
        # for i in range(self.count_paths.shape[0]):
        #     for j in range(self.count_paths.shape[1]):
        #         n = self.count_paths[i][j]
        #         if(n == 0):
        #             continue
        #         weight = [self.edges.get(f'{i}_{j}_{k}')[0] for k in range(n)]
        #         self.edge_weight.append(sum(weight))
        #         self.edge_index.append([i , j])
        #         self.edge_attribute.extend( [self.edges.get(f'{i}_{j}_{k}') for k in range(n)] )
        import torch
        self.edge_index = torch.tensor(self.edge_index, dtype= torch.long).reshape(2, -1)
        self.edge_attribute = torch.tensor(self.edge_attribute, dtype= torch.long)


    def read_data(self):
        with open(self.file, "r") as f:
            lines = f.readlines()
            #get num_nodes and num_domains from the first line
            line0 = lines[0].split()
            self.num_nodes = int(line0[0])
            self.num_domains = int(line0[1])
            count_paths = np.zeros((self.num_nodes, self.num_nodes)).astype(np.int64)
            #edges is a dictionary with key: s_t_k and value is a list with size 2 
            # self.normal_edges = {}
            #get source and target from the seconde line
            self.edges = nb.typed.Dict().empty(
                key_type= nb.types.unicode_type,
                value_type= nb.typeof((0, 0)),
            )
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
        left_domains = [False for _ in range(num_domains + 1)]
        visited_vertex = [False for _ in range(num_nodes)]
    
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
                key = '_'.join([str(curr), str(t), str(k)])
                d = edges[key][1]
                if left_domains[d]:
                    continue
                cost += edges[key][0]
                left_domains[d] = True
                curr = t
                stop = False
                break
            if stop:
                return MAX_INT
        return cost
        
    def __call__(self, gene: np.ndarray):
        # decode
        idx = np.sort(np.argsort(gene[0])[:self.dim])
        
        # idx = np.arange(self.dim)
        gene = np.ascontiguousarray(gene[:, idx])
        cost = __class__.func(gene, self.source, self.target,
                                self.num_nodes, self.num_domains, self.edges, self.count_paths)
        return cost
