from typing import Tuple, Type
import numpy as np
from ..tasks.task import AbstractTask
from ..EA import Individual, Population
from .Search import *
import numba as nb

class AbstractCrossover():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pa: Individual, pb: Individual, skf_oa= None, skf_ob= None, *args, **kwargs) -> Tuple[Individual, Individual]:
        pass
    def getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed = None):
        self.dim_uss = max([t.dim for t in tasks])
        self.nb_tasks = len(tasks)
        self.tasks = tasks
        self.IndClass = IndClass
        #seed
        np.random.seed(seed)
        pass
    
    def update(self, *args, **kwargs) -> None:
        pass

class NoCrossover(AbstractCrossover):
    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        oa = self.IndClass(None, self.dim_uss)
        ob = self.IndClass(None, self.dim_uss)

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob
        

class SBX_Crossover(AbstractCrossover):
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nc = 15, *args, **kwargs):
        self.nc = nc

    def __call__(self, pa: Individual, pb: Individual, skf_oa= None, skf_ob= None, *args, **kwargs) -> Tuple[Individual, Individual]:
        u = np.random.rand(self.dim_uss)

        # ~1 TODO
        beta = np.where(u < 0.5, (2*u)**(1/(self.nc +1)), (2 * (1 - u))**(-1 / (1 + self.nc)))

        #like pa
        oa = self.IndClass(np.clip(0.5*((1 + beta) * pa.genes + (1 - beta) * pb.genes), 0, 1))
        #like pb
        ob = self.IndClass(np.clip(0.5*((1 - beta) * pa.genes + (1 + beta) * pb.genes), 0, 1))

        if pa.skill_factor == pb.skill_factor:
            idx_swap = np.where(np.random.rand(self.dim_uss) < 0.5)[0]
            oa.genes[idx_swap], ob.genes[idx_swap] = ob.genes[idx_swap], oa.genes[idx_swap]

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob

class newSBX(AbstractCrossover):
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nc = 2, gamma = .9, alpha = 6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nc = nc
        self.gamma = gamma
        self.alpha = alpha
    
    def getInforTasks(self,  IndClass: Type[Individual], tasks: list[AbstractTask], seed = None):
        super().getInforTasks(IndClass, tasks, seed= seed)
        self.prob = np.zeros((self.nb_tasks, self.nb_tasks, self.dim_uss)) + 0.5
        for i in range(self.nb_tasks):
            self.prob[i, i, :] = 1

        # nb all offspring borned by crossover at each dimension by tasks
        self.count_crossover_each_dimensions = np.zeros((self.nb_tasks, self.nb_tasks, self.dim_uss))
        # nb inds alive after epoch
        self.sucess_crossover_each_dimensions = np.zeros((self.nb_tasks, self.nb_tasks, self.dim_uss))

        # id and idx_crossover of epoch
        self.epoch_id = []
        self.epoch_idx_crossover = []

        # skillfactor of parents
        self.skf_parent = []
        
    def update(self, population: Population):

        # calculate sucess_crossover_each_dimensions
        for subPop in population:
            for ind in subPop:
                try:
                    index = self.epoch_id.index(id(ind))
                except:
                    continue
                self.sucess_crossover_each_dimensions[self.skf_parent[index][0], self.skf_parent[index][1]] += \
                    self.epoch_idx_crossover[index]

        # percent success: per_success = success / count
        per_success = (
            self.sucess_crossover_each_dimensions / (self.count_crossover_each_dimensions + 1e-10)
        )** (1/self.alpha)

        # new prob
        new_prob = np.copy(per_success)
        # prob_success greater than or equal intra -> p = 1
        tmp_smaller_intra_change = np.empty_like(self.count_crossover_each_dimensions)
        for i in range(self.nb_tasks):
            tmp_smaller_intra_change[i] = (per_success[i] <= per_success[i, i])
        
        new_prob = np.where(
            tmp_smaller_intra_change,
            new_prob,
            1
        )
        new_prob = np.where(
            self.count_crossover_each_dimensions != 0,
            new_prob,
            self.prob
        )

        # update prob
        self.prob = self.prob * self.gamma + (1 - self.gamma) * new_prob
        self.prob = np.clip(self.prob, 1/self.dim_uss, 1)

        # reset
        self.count_crossover_each_dimensions = np.zeros_like(self.count_crossover_each_dimensions)
        self.sucess_crossover_each_dimensions = np.zeros_like(self.sucess_crossover_each_dimensions)
        self.epoch_id = []
        self.epoch_idx_crossover = []
        self.skf_parent = []

    def __call__(self, pa: Individual, pb: Individual, skf_oa= None, skf_ob= None, *args, **kwargs) -> Tuple[Individual, Individual]:
        if skf_oa == pa.skill_factor:
            p_of_oa = pa
        elif skf_oa == pb.skill_factor:
            p_of_oa = pb
        else:
            raise ValueError()
        if skf_ob == pa.skill_factor:
            p_of_ob = pa
        elif skf_ob == pb.skill_factor:
            p_of_ob = pb
        else:
            raise ValueError()

        u = np.random.rand(self.dim_uss)
        
        # ~1
        beta = np.where(u < 0.5, (2*u)**(1/(self.nc +1)), (2 * (1 - u))**(-1 / (self.nc + 1)))

        if pa.skill_factor == pb.skill_factor:
            idx_crossover = np.ones((self.dim_uss, ))

            #like pa
            oa = self.IndClass(np.clip(0.5*((1 + beta) * pa.genes + (1 - beta) * pb.genes), 0, 1))
            #like pb
            ob = self.IndClass(np.clip(0.5*((1 - beta) * pa.genes + (1 + beta) * pb.genes), 0, 1))

            #swap
            idx_swap = np.where(np.random.rand(len(pa)) < 0.5)[0]
            oa.genes[idx_swap], ob.genes[idx_swap] = ob.genes[idx_swap], oa.genes[idx_swap]
        else:
            idx_crossover = np.random.rand(self.dim_uss) < self.prob[pa.skill_factor, pb.skill_factor]

            if np.all(idx_crossover == 0) or np.all(pa[idx_crossover] == pb[idx_crossover]):
                # alway crossover -> new individual
                idx_notsame = np.where(pa.genes != pb.genes)[0].tolist()
                if len(idx_notsame) == 0:
                    idx_crossover = np.ones((self.dim_uss, ))
                else:
                    idx_crossover[np.random.choice(idx_notsame)] = 1

            #like pa
            oa = self.IndClass(np.where(idx_crossover, np.clip(0.5*((1 + beta) * pa.genes + (1 - beta) * pb.genes), 0, 1), p_of_oa))
            #like pb
            ob = self.IndClass(np.where(idx_crossover, np.clip(0.5*((1 - beta) * pa.genes + (1 + beta) * pb.genes), 0, 1), p_of_ob))

            #swap
            if skf_ob == skf_oa:
                idx_swap = np.where(np.random.rand(len(pa)) < 0.5)[0]
                oa.genes[idx_swap], ob.genes[idx_swap] = ob.genes[idx_swap], oa.genes[idx_swap]
        
        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob

        self.count_crossover_each_dimensions[pa.skill_factor, pb.skill_factor] += 2 * idx_crossover

        self.skf_parent.append([pa.skill_factor, pb.skill_factor])
        self.skf_parent.append([pa.skill_factor, pb.skill_factor])
        self.epoch_id.append(id(oa))
        self.epoch_id.append(id(ob))
        self.epoch_idx_crossover.append(idx_crossover)
        self.epoch_idx_crossover.append(idx_crossover)

        return oa, ob

@nb.njit
def pmx_func(p1, p2, t1, t2,  dim_uss):
    oa = np.empty_like(p1)
    ob = np.empty_like(p1)
    
    mid = np.copy(p2[t1:t2])
    mid_b = np.copy(p1[t1 : t2])
    
    added = np.zeros_like(p1)
    added_b = np.zeros_like(p2)
    
    added[mid] = 1
    added_b[mid_b] = 1
    
    redundant_idx = []
    redundant_idx_b = []
    
    
    for i in range(t1):
        if added[p1[i]]:
            redundant_idx.append(i)
        else:
            oa[i] = p1[i]
            added[oa[i]] = 1
            
        if added_b[p2[i]]:
            redundant_idx_b.append(i)
        else:
            ob[i] = p2[i]
            added_b[ob[i]] = 1
            
    for i in range(t2, dim_uss):
        
        if added[p1[i]]:
            redundant_idx.append(i)
        else:
            oa[i] = p1[i]
            added[oa[i]] = 1
            
        if added_b[p2[i]]:
            redundant_idx_b.append(i)
        else:
            ob[i] = p2[i]
            added_b[ob[i]] = 1
    
    redundant = np.empty(len(redundant_idx))
    redundant_b = np.empty(len(redundant_idx_b))
    
    cnt = 0
    cnt_b = 0
    
    for i in range(t1, t2):
        if added[p1[i]] == 0:
            redundant[cnt] = p1[i]
            cnt+=1
        if added_b[p2[i]] == 0:
            redundant_b[cnt_b] = p2[i]
            cnt_b+=1
    
    redundant_idx = np.array(redundant_idx)
    redundant_idx_b = np.array(redundant_idx_b)
    
    oa[redundant_idx] = redundant
    ob[redundant_idx_b] = redundant_b
    
    oa[t1:t2] = mid
    ob[t1:t2] = mid_b
    return oa, ob
    

class PMX_Crossover(AbstractCrossover):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        t1 = np.random.randint(0, self.dim_uss + 1)
        t2 = np.random.randint(t1, self.dim_uss + 1)
        genes_oa, genes_ob = pmx_func(pa.genes, pb.genes, t1, t2, self.dim_uss)
        oa = self.IndClass(genes_oa)
        ob = self.IndClass(genes_ob)
        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob

        

class TPX_Crossover(AbstractCrossover):
    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        t1, t2 = np.random.randint(0, self.dim_uss + 1, 2)
        if t1 > t2:
            t1, t2 = t2, t1

        genes_oa = np.copy(pa.genes)
        genes_ob = np.copy(pb.genes)

        genes_oa[t1:t2], genes_ob[t1:t2] = genes_ob[t1:t2], genes_oa[t1:t2]

        oa = self.IndClass(genes_oa)
        ob = self.IndClass(genes_ob)

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob


class IDPCEDU_Crossover(AbstractCrossover):
    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        genes_oa, genes_ob = np.empty_like(pa), np.empty_like(pb)

        #PMX
        t1, t2 = np.random.randint(0, self.dim_uss + 1, 2)
        if t1 > t2:
            t1, t2 = t2, t1
        genes_oa[0], genes_ob[0] = pmx_func(pa.genes[0], pb.genes[0], t1, t2, self.dim_uss)

        #TPX
        t1, t2 = np.random.randint(0, self.dim_uss + 1, 2)
        if t1 > t2:
            t1, t2 = t2, t1
        genes_oa[1], genes_ob[1] = pa.genes[1], pb.genes[1]
        genes_oa[1, t1:t2], genes_ob[1, t1:t2] = genes_ob[1, t1:t2], genes_oa[1, t1:t2]

        oa = self.IndClass(genes_oa)
        ob = self.IndClass(genes_ob)

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob