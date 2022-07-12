#!/usr/bin/env python
# coding: utf-8

# In[21]:


from MFEA_lib.model import MFEA_base, SA_MFEA, SMP_MFEA, MFEA1
from MFEA_lib.model.utils import *
from MFEA_lib.operators.Crossover import *
from MFEA_lib.operators.Mutation import *
from MFEA_lib.operators.Selection import *


# In[22]:


import sys
MAX_INT = sys.maxsize
class TULKH_FUNC(AbstractTask):    
    def __init__(self, path_data):
        self.datas = {}
        self.N: int
        self.e: np.ndarray
        self.l: np.ndarray
        self.d: np.ndarray
        self.path_data = path_data
        self.name = path_data
        self.read_data()
            
    def read_data(self):
        with open(self.path_data, 'r') as file: 
            n = int(file.readline())
            early = []
            late = []
            delay = []
            cost = []
            time = []
            for _ in range(n+1):
                e, l, d = file.readline().strip().split(' ')
                early.append(int(e))
                late.append(int(l))
                delay.append(int(d))
            for _ in range(n+1):
                costi = [int(a) for a in file.readline().strip().split(' ')]
                cost.append(costi)
            for _ in range(n+1):
                timei = [int(a) for a in file.readline().strip().split(' ')]
                time.append(timei)
        data = {}
        data['n'] = n
        data['early'] = np.array(early)
        data['late'] = np.array(late)
        data['delay'] = np.array(delay)
        data['cost'] = np.array(cost)
        data['time'] = np.array(time)
        self.dim = n
        self.data = data


    @staticmethod
    def func(gene,
             data,
             ):
        cost = 0 
        idx = (np.argsort(-gene) + 1).tolist()
        curr = 0
        curr_time = 0
        # visited_vertex = [False for i in range(data['n'] + 1)]
        # visited_vertex[0] = True
        for _ in range(data['n']):
            stop = True
            for i, t in enumerate(idx):
                # if visited_vertex[t]:
                #     continue
                arrive_time = curr_time + data['time'][curr][t]
                if arrive_time > data['late'][t]:
                    continue
                
                assert t != curr and data['cost'][curr][t] > 0

                cost += data['cost'][curr][t]
                curr = t
                cur_time = max(arrive_time, data['early'][t])
                cur_time += data['delay'][t]
                # visited_vertex[t] = True
                
                stop = False
                idx.pop(i)
                break
                
            if stop:
                return MAX_INT
        assert data['cost'][curr][0] > 0
        assert len(idx) == 0
        return cost + data['cost'][curr][0]
        
    def __call__(self, gene: np.ndarray):
        # decode
        gene = gene[:self.dim]
        # eval
        return __class__.func(gene, self.data)


# In[23]:


from tqdm import tqdm
class Ind_TULKH(Individual):
    def __init__(self, genes, dim=None) -> None:
        super().__init__(genes, dim)
        if genes is None:
            self.genes: np.ndarray = np.random.permutation(dim)  + 1
            
class TULKH_benchmark:
    def get_tasks():
        print('\rReading data...')
        tasks = []
        file_list = ['DATA/' + file_name for file_name in ['50points_5days_DoubleDiscrepancy.txt', '10points_1day.txt', '25points_3days.txt']]
        for file_name in tqdm(file_list):
            tasks.append(TULKH_FUNC(file_name))
                         
        return tasks, Ind_TULKH


# In[24]:


tasks, IndClass = TULKH_benchmark.get_tasks()


# In[25]:


class TULKH_Crossover(AbstractCrossover):
    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        genes_oa, genes_ob = np.empty_like(pa), np.empty_like(pb)

        #PMX
        t1, t2 = np.random.randint(0, self.dim_uss + 1, 2)
        if t1 > t2:
            t1, t2 = t2, t1
        genes_oa, genes_ob = pmx_func(pa.genes - 1, pb.genes - 1, t1, t2, self.dim_uss)

        oa = self.IndClass(genes_oa + 1)
        ob = self.IndClass(genes_ob + 1)
        assert np.amin(genes_oa) >= 0 and np.amin(genes_ob) >= 0, f'{pa.genes} loi day nay'
        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob

class TULKH_Mutation(AbstractMutation):
    def getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)
        
    def __call__(self, ind: Individual, return_newInd: bool, *arg, **kwargs) -> Individual:
        return ind


# In[26]:


baseModel = MFEA_base.model()
baseModel.compile(
    IndClass= IndClass,
    tasks= tasks,
    # crossover= newSBX(nc = 2, gamma= 0.4, alpha= 6),
    crossover= TULKH_Crossover(),
    mutation= TULKH_Mutation(),
    selection= ElitismSelection()
)
solve = baseModel.fit(
    nb_generations = 1000, rmp = 0.3, nb_inds_each_task= 100, 
    bound_pop= [0, 1], evaluate_initial_skillFactor= True
)
print(solve)
baseModel.render_history()