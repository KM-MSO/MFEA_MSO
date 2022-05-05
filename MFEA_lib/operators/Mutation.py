from copy import deepcopy
from typing import Deque, Tuple, Type
from matplotlib import pyplot as plt
import numpy as np

from ..tasks.task import AbstractTask
from ..EA import Individual, Population, SubPopulation

class AbstractMutation():
    def __init__(self, *arg, **kwargs):
        self.pm = None
        pass
    def __call__(self, ind: Individual, return_newInd:bool, *arg, **kwargs) -> Individual:
        pass
    def getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed = None):
        self.dim_uss = max([t.dim for t in tasks])
        self.nb_tasks = len(tasks)
        if self.pm is None:
            self.pm = 1/self.dim_uss
        self.tasks = tasks
        self.IndClass = IndClass
        #seed
        np.random.seed(seed)
        pass
    def update(self, *arg, **kwargs) -> None:
        pass
    def compute_accept_prob(self, new_fcost, min_fcost): 
        pass
class NoMutation(AbstractMutation):
    def __call__(self, ind: Individual, return_newInd:bool, *arg, **kwargs) -> Individual:
        if return_newInd:
            newInd =  self.IndClass(genes= np.copy(ind.genes))
            newInd.skill_factor = ind.skill_factor
            newInd.fcost = ind.fcost
            return newInd
        else:
            return ind
    
class Polynomial_Mutation(AbstractMutation):
    '''
    p in [0, 1]^n
    '''
    def __init__(self, nm = 15, pm = None, *arg, **kwargs):
        '''
        nm: parameters of Polynomial_mutation
        pm: prob mutate of Polynomial_mutation
        '''
        super().__init__(*arg, **kwargs)
        self.nm = nm
        self.pm = pm

    def __call__(self, ind: Individual, return_newInd:bool, *arg, **kwargs) -> Individual:
        idx_mutation = np.where(np.random.rand(self.dim_uss) <= self.pm)[0]

        #NOTE 
        u = np.zeros((self.dim_uss,)) + 0.5
        u[idx_mutation] = np.random.rand(len(idx_mutation))

        delta = np.where(u < 0.5,
            # delta_l
            (2*u)**(1/(self.nm + 1)) - 1,
            # delta_r
            1 - (2*(1-u))**(1/(self.nm + 1))
        )

        if return_newInd:
            newInd = self.IndClass(
                genes= np.where(delta < 0,
                    # delta_l: ind -> 0
                    ind.genes + delta * ind.genes,
                    # delta_r: ind -> 1
                    ind.genes + delta * (1 - ind.genes)
                )
            )
            newInd.skill_factor = ind.skill_factor
            return newInd
        else:
            ind.genes = np.where(delta < 0,
                    # delta_l: ind -> 0
                    ind.genes + delta * ind.genes,
                    # delta_r: ind -> 1
                    ind.genes + delta * (1 - ind.genes)
            )
            ind.fcost = None

            return ind
    
class GaussMutation(AbstractMutation):
    '''
    p in [0, 1]^n
    '''
    def __init__(self, scale = 0.1, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.scale = scale
    
    def __call__(self, ind: Individual, return_newInd:bool, *arg, **kwargs) -> Individual:
        idx_mutation = np.where(np.random.rand(self.dim_uss) <= self.pm)[0]
        
        mutate_genes = ind[idx_mutation] + np.random.normal(0, self.scale, size = len(idx_mutation))

        # clip 0 1
        idx_tmp = np.where(mutate_genes > 1)[0]
        mutate_genes[idx_tmp] = 1 - np.random.rand(len(idx_tmp)) * (1 - ind[idx_mutation][idx_tmp])
        idx_tmp = np.where(mutate_genes < 0)[0]
        mutate_genes[idx_tmp] = ind[idx_mutation][idx_tmp] * np.random.rand(len(idx_tmp))
        
        if return_newInd:
            new_genes = np.copy(ind.genes)
            new_genes[idx_mutation] = mutate_genes
            newInd = self.IndClass(genes= new_genes)
            newInd.skill_factor = ind.skill_factor
            return newInd
        else:
            ind.genes[idx_mutation] = mutate_genes
            ind.fcost = None
            return ind

class PMD_Scale(AbstractMutation):
    def __init__(self, nm = 5, lenMem = 30, default_scale = 0.5, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.lenMem = lenMem
        self.default_scale = default_scale

    def getInforTasks(self, tasks: list[AbstractTask], seed=None):
        super().getInforTasks(tasks, seed)

    def __call__(self, ind: Individual, return_newInd: bool, *arg, **kwargs) -> Individual:
        return super().__call__(ind, return_newInd, *arg, **kwargs)

class IDPCEDU_Mutation(AbstractMutation):
    def getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)
        self.S_tasks = [np.amax(t.count_paths, axis= 1)  for t in tasks]
        

    def __call__(self, ind: Individual, return_newInd: bool, *arg, **kwargs) -> Individual:
        if return_newInd:
            new_genes:np.ndarray = np.copy(ind.genes)

            i, j = np.random.randint(0, self.dim_uss, 2)
            new_genes[0, i], new_genes[0, j] = new_genes[0, j], new_genes[0, i]
            i = np.random.randint(0, len(self.S_tasks[ind.skill_factor]))
            new_genes[1, i] = np.random.randint(0, self.S_tasks[ind.skill_factor][i])
            newInd = self.IndClass(genes= new_genes)
            newInd.skill_factor = ind.skill_factor
            return newInd
        else:
            i, j = np.random.randint(0, self.dim_uss, 2)
            ind.genes[0, i], ind.genes[0, j] = ind.genes[0, j], ind.genes[0, i]
            i = np.random.randint(0, len(self.S_tasks[ind.skill_factor]))
            ind.genes[1, i] = np.random.randint(0, self.S_tasks[ind.skill_factor][i])

            return ind

class Directional_Mutation(AbstractMutation):

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def  getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed = None):
        super().getInforTasks(IndClass,tasks, seed)
        self.direction = [True]*self.nb_tasks
        self.prev_mean = [float('inf')]*self.nb_tasks

    def __call__(self, ind: Individual,return_newInd : bool, *arg, **kwargs) -> Individual:
        r = np.random.rand()
        beta1 = np.exp(r ** 2) * np.exp(r - 2/r)
        beta2 = np.exp(r - r ** 2) *np.exp(r - 2/r)

        if(np.random.rand() < 0.5):
            if self.direction[ind.skill_factor] is True : 
                newInd = self.IndClass(genes= ind.genes + beta1 * (1 - ind.genes))
                newInd.skill_factor = ind.skill_factor
            else:
                newInd = self.IndClass(genes= np.clip(ind.genes - beta1 * (1 - ind.genes), 0, 1))
                newInd.skill_factor = ind.skill_factor
        else:
            if self.direction[ind.skill_factor] is True : 
                newInd = self.IndClass(genes= ind.genes - beta2 * ind.genes)
                newInd.skill_factor = ind.skill_factor
            else:
                newInd = self.IndClass(genes= np.clip(ind.genes + beta2 * ind.genes, 0, 1))
                newInd.skill_factor = ind.skill_factor
        if return_newInd:
            return newInd
        else:
            ind.genes = newInd.genes
            ind.fcost = newInd.fcost
            return ind

    def update(self, population:Population):
        idx = 0
        for subpop in population.ls_subPop:
            curr_mean = np.mean([ind.fcost for ind in subpop.ls_inds])
            if curr_mean > self.prev_mean[idx]:
                self.direction[idx] = False
            self.prev_mean[idx] = curr_mean
            idx += 1 
        # print(self.direction)
         