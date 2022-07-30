#!/usr/bin/env python
# coding: utf-8

# In[21]:


from dataclasses import replace
from MFEA_lib.model import MFEA_base, SA_MFEA, SMP_MFEA, MFEA1
from MFEA_lib.model.utils import *
from MFEA_lib.operators.Crossover import *
from MFEA_lib.operators.Mutation import *
from MFEA_lib.operators.Selection import *
from scipy.stats import rankdata

# In[22]:


import sys
MAX_INT = sys.maxsize
class TULKH_FUNC(AbstractTask):    
    
    def __init__(self, path_data, objective):
        self.datas = {}
        self.N: int
        self.e: np.ndarray
        self.l: np.ndarray
        self.d: np.ndarray
        self.path_data = path_data
        self.name = path_data
        self.objective = objective
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
        data['objective'] = self.objective
        if self.objective == 'mutate':
            idx = np.where(np.random.rand(data['cost'].shape[0], data['cost'].shape[1]) > 0.5, True, False)
            data['cost'][idx] = data['cost'][idx] * 2 
        # data['prior'] = rankdata(data['late'])
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

        for curr_idx in range(data['n']):
            stop = True
            for i, t in enumerate(idx):
            
                
                arrive_time = curr_time + data['time'][curr][t]
                
                if arrive_time > data['late'][t]:
                    continue
                    
                assert t != curr and data['cost'][curr][t] > 0

                cost += data['cost'][curr][t]
                curr = t
                
                curr_time = max(arrive_time, data['early'][t])
                curr_time += data['delay'][t]
                
                stop = False
                idx.pop(i)
                break
                
            if stop:
                if data['objective'] == 'cost':
                # print(curr_idx / data['n'])
                # print(curr_idx, data['n'])
                    return MAX_INT
                return -curr_idx 
            
        assert data['cost'][curr][0] > 0
        assert len(idx) == 0
        if data['objective'] == 'cost':
            return cost + data['cost'][curr][0]
        else:
            return - data['n'] - 1000 / cost
        
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
        print('\rReading data...', )
        tasks = []
        file_list = ['DATA/' + file_name for file_name in ['50points_5days_DoubleDiscrepancy.txt']]
        for i in range(1):
            for file_name in tqdm(file_list):
                # tasks.append(TULKH_FUNC(file_name, objective = 'cost'))
                tasks.append(TULKH_FUNC(file_name, objective = 'time'))
                tasks.append(TULKH_FUNC(file_name, objective = 'time'))
                tasks.append(TULKH_FUNC(file_name, objective = 'time'))
                tasks.append(TULKH_FUNC(file_name, objective = 'time'))
                tasks.append(TULKH_FUNC(file_name, objective = 'time'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                tasks.append(TULKH_FUNC(file_name, objective = 'mutate'))
                
                         
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
        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob

class TULKH_Mutation(AbstractMutation):
    def getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)
        
    def __call__(self, ind: Individual, return_newInd: bool, *arg, **kwargs) -> Individual:
        if return_newInd:
            new_genes:np.ndarray = np.copy(ind.genes)

            i, j = np.random.randint(0, self.dim_uss, 2)
            new_genes[i], new_genes[j] = new_genes[j], new_genes[i]
            newInd = self.IndClass(genes= new_genes)
            newInd.skill_factor = ind.skill_factor
            return newInd
        else:
            i, j = np.random.randint(0, self.dim_uss, 2)
            ind.genes[i], ind.genes[j] = ind.genes[j], ind.genes[i]
            return ind


# In[26]:
rs = []
for i in range(1):
    baseModel = MFEA_base.model()
    baseModel.compile(
        IndClass= IndClass,
        tasks= tasks,
        # crossover= newSBX(nc = 2, gamma= 0.4, alpha= 6),
        crossover= TULKH_Crossover(),
        mutation= TULKH_Mutation(),
        selection= ElitismSelection(p =0.3)
    )
    solve = baseModel.fit(
        nb_generations = 2000, rmp = 0.1, nb_inds_each_task= 100, 
        bound_pop= [0, 1], evaluate_initial_skillFactor= True
    )
    print(solve)
    rs.append([solve, baseModel.history_cost])

import pickle
with open('rs.pickle', 'wb') as file: 
    pickle.dump(rs, file)