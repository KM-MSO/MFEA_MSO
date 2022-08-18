import numpy as np
import numba as nb
import sys
import copy
from queue import PriorityQueue
MAX_INT = sys.maxsize
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

#----------------------------------------------------------------------------------------------------------------------------
#a solution is an permutation start from 0 to n - 1, k is also counted from 0 but domain is counted from 1
class IDPC_EDU_FUNC(AbstractTask):    
    def __init__(self, dataset_path, file_name):
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
        self.read_data()
            
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
    @nb.njit(
        nb.int64(
            nb.typeof(np.array([[1]]).astype(np.int64)),
            nb.int64,
            nb.int64,
            nb.int64,
            nb.int64,
            nb.typeof(nb.typed.Dict().empty(
                key_type= nb.types.unicode_type,
                value_type= nb.typeof((0, 0)),
            )),
            nb.typeof(np.array([[1]]).astype(np.int64)),
        )
    )
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
        return __class__.func(gene, self.source, self.target,
                         self.num_nodes, self.num_domains, self.edges, self.count_paths)

class Edge :
    def __init__(self, startNode, endNode, domain, weight, visitedNode) :
        self.startNode = startNode
        self.endNode = endNode
        self.domain = domain
        self.weight = weight
        self.visitedNode = visitedNode

    def __add__(self,other): 
        assert self.endNode == other.startNode
        assert self.domain == other.domain
        visitedNode = []
        for node in self.visitedNode :
            visitedNode.append(node)
        for node in other.visitedNode :
            if node not in self.visitedNode :
                visitedNode.append(node)
        weight = self.weight + other.weight
        return Edge (self.startNode, other.endNode, self.domain, weight, visitedNode)

    def __gt__ (self, other) :
        if (self.weight > other.weight) :
            return True
        else :
            if self.weight == other.weight and len(self.visitedNode) > len(other.visitedNode) :
                return True
            else :
                return False
def intersection (list1, list2) :
    for i in list1 :
        if i in list2 : 
            return True
    return False
class IDPC_EDU(AbstractTask):
    def __init__(self, dataset_path, file_name):
        self.file = str(dataset_path) + '/'  + file_name
        self.adj1: list 
        self.start: int
        self.end: int
        self.numDomains: int
        self.numNodes: int
        self.num_edges: int = int(file_name[:-5].split('x')[-1])
        self.edges: np.ndarray
        self.dim: int
        self.name = file_name.split('.')[0]
        self.read_data()
    def read_data(self):
        with open(self.file,'r') as f:
            lines = f.readlines()
            tmp = lines[0].split()
            self.numNodes = int(tmp[0])
            self.numDomains = int(tmp[1])
            self.dim = self.numNodes+self.numDomains
            weight_matrix = np.ones([self.numNodes, self.numNodes, self.numDomains]) * -1
            tmp = lines[1].split()
            self.start = int(tmp[0]) - 1
            self.end = int(tmp[1]) - 1
            self.adj1 = []
            self.node_to_domain = []
            for i in range(self.numNodes) :
                tmp = []
                self.node_to_domain.append(tmp)
            for i in range(self.numNodes) :
                adj_temp = []
                for j in range(self.numNodes) :
                    domain_list = [1] * self.numDomains
                    adj_temp.append(domain_list)
                self.adj1.append(adj_temp)

            adj2 = []
            for i in range(self.numDomains) :
                node_list = []
                for i in range(self.numNodes) :
                    tmp = []
                    node_list.append(tmp)
                adj2.append(node_list)

            for i in range(2,len(lines)) :
                tmp = lines[i].split()
                weight = int(tmp[2])
                domain = int(tmp[3]) - 1
                startNode = int(tmp[0]) - 1
                endNode = int(tmp[1]) - 1

                # weight[startNode, endNode, domain] = weight
                if startNode == self.end or endNode == self.start :
                    continue
                edge = Edge (startNode, endNode, domain, weight, visitedNode= [])
                adj2[domain][startNode].append(edge)
                if domain not in self.node_to_domain :
                    self.node_to_domain[startNode].append(domain)
        for domain in range(self.numDomains) :
            adj_domain = adj2[domain]
            for i in range(self.numNodes) :
                check = [True] * self.numNodes
                priority_queue = PriorityQueue()
                for edge in adj_domain[i] :
                    priority_queue.put(edge)
                if priority_queue.empty() == False :
                    edge = priority_queue.get()
                    self.adj1[edge.startNode][ edge.endNode][edge.domain] = edge
                while priority_queue.empty() == False :
                    next_Node = edge.endNode
                    check[next_Node] = False
                    for candidate in adj_domain[next_Node] :
                        try :
                            priority_queue.put(edge + candidate)
                        except :
                            print("___________________")
                            print(edge.endNode)
                            for c in adj_domain[next_Node] :
                                print(c.startNode)

                    edge = priority_queue.get()
                    while check[edge.endNode] == False and priority_queue.empty() == False :
                        edge = priority_queue.get()
                    if check[next_Node] == True :
                        self.adj1[edge.startNode][ edge.endNode][edge.domain] = edge
    def find_next_domain (self,priority_list, visited_list, node_to_domain) :
        priority_list_tmp = copy.deepcopy(priority_list)
        for i in visited_list :
            priority_list_tmp[i] = -1
        for i in range(self.numDomains) :
            if i not in node_to_domain :
                priority_list_tmp[i] = -1
        return np.argmax(priority_list_tmp)
    def find_next_Node (self,priority_list, visited_list, domain, current_Node) :
        priority_list_tmp = copy.deepcopy(priority_list)
        for i in visited_list :
            priority_list_tmp[i] = -1
        candidate_Node = np.argmax(priority_list_tmp)
        stop = False
        if self.adj1[current_Node][candidate_Node][domain] == 1 :
            stop = True
        else :
            visited_Node = self.adj1[current_Node][candidate_Node][domain].visitedNode
            if intersection(visited_list, visited_Node) == True :
                stop = True
        while stop == True :
            priority_list_tmp[candidate_Node] = -1
            if np.sum(priority_list_tmp) < -(self.numNodes - 1) :
                return None
            candidate_Node = np.argmax(priority_list_tmp)
            stop = False
            if self.adj1[current_Node][candidate_Node][domain] == 1 :
                stop = True
            else :
                visited_Node = self.adj1[current_Node][candidate_Node][domain].visitedNode
                if intersection(visited_list, visited_Node) == True :
                    stop = True
        return candidate_Node
    def func(genes,start,end,numDomains,numNodes,find_next_domain,find_next_Node,node_to_domain,adj1) :
        visitedNodes = []
        visitedDomain = []
        genes_domain = genes[0:numDomains]
        genes_node  = genes[numDomains : (numNodes+ numDomains)]
        current_Domain = 0
        current_Node = start
        distance = 0
        residual = 1000000
        while current_Node != end :
            current_Domain = find_next_domain (genes_domain, visitedDomain, node_to_domain[current_Node])
            next_Node = find_next_Node (genes_node, visitedNodes, current_Domain, current_Node)

            if adj1[current_Node][end][current_Domain] != 1 :
                residual = min (residual, distance + adj1[current_Node][end][current_Domain].weight)

            if next_Node == None :
                return residual

            distance += adj1[current_Node][next_Node][current_Domain].weight
            visitedNodes.append(next_Node)
            visitedNodes += adj1[current_Node][next_Node][current_Domain].visitedNode
            visitedDomain.append(current_Domain)
            current_Node = next_Node
        return min(distance, residual)

    def __call__(self, gene: np.ndarray):
        # decode
        
        # eval
        return __class__.func(gene, self.start, self.end,
                         self.numDomains, self.numNodes, self.find_next_domain, self.find_next_Node,self.node_to_domain,self.adj1)


class IDPC_EDU_new(AbstractTask):    
    def __init__(self, dataset_path, file_name):
        self.file = str(dataset_path) + '/'  + file_name
        self.datas = {}
        self.source: int
        self.target: int
        self.num_domains: int
        self.num_nodes: int
        self.num_edges: int = int(file_name[:-5].split('x')[-1])
        self.edges: dict ={}
        self.dim: int 
        self.name = file_name.split('.')[0]
        self.read_data()
            
    def read_data(self):
        with open(self.file, "r") as f:
            lines = f.readlines()
            #get num_nodes and num_domains from the first line
            line0 = lines[0].split()
            self.num_nodes = int(line0[0])
            self.num_domains = int(line0[1])
            #get source and target from the seconde line
            line1 = lines[1].split()
            self.source = int(line1[0])
            self.target = int(line1[1])
            #get all edges
            lines = lines[2:]
            for line in lines:
                data = [int(x) for x in line.split()]
                if data[0] not in self.edges.keys(): 
                    self.edges[data[0]] = [data[1:]]
                else:
                    self.edges[data[0]].append(data[1:])
            self.dim = self.num_nodes
    def func(gene,
             source,
             target,
             num_nodes,
             num_domains,
             edges,
             ):
        cost =0 
        leftDomain = -1
        domain = np.zeros(num_domains+1)
        nodes = np.zeros(num_nodes+1)
        currNode=  source
        for idx in range(len(gene)):
            if currNode == target:
                return cost
            l = int(gene[idx]*len(edges[currNode]))
            if l == len(edges[currNode]):
                l = l-1
            nextNode,w,d = edges[currNode][l]
            # print(nextNode)
            if nodes[nextNode] == 1:
                return 99999
            if domain[d] == 1 and d != leftDomain:
                return 99999
            nodes[nextNode] = 1
            domain[d] = 1
            leftDomain = d
            currNode = nextNode
            cost += w

        
    def __call__(self, gene: np.ndarray):
        # decode
        # idx = np.argsort(gene[0])[:self.dim]
        # eval
        return __class__.func(gene, self.source, self.target,
                         self.num_nodes, self.num_domains, self.edges)