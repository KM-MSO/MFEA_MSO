from typing import Tuple, Type
import numpy as np
import scipy.stats

from ..tasks.task import AbstractTask
from ..EA import Individual, Population

class AbstractSearch():
    def __init__(self) -> None:
        pass
    def __call__(self, *args, **kwargs) -> Individual:
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

class SHADE(AbstractSearch):
    def __init__(self, len_mem = 30, p_best_type:str = 'ontop', p_ontop = 0.1, tournament_size = 2) -> None:
        '''
        `p_best_type`: `random` || `tournament` || `ontop`
        '''
        super().__init__()
        self.len_mem = len_mem
        self.p_best_type = p_best_type
        self.p_ontop = p_ontop
        self.tournament_size = tournament_size

    def getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed = None):
        super().getInforTasks(IndClass, tasks, seed= seed)
        # memory of cr and F
        self.M_cr = np.zeros(shape = (self.nb_tasks, self.len_mem, ), dtype= float) + 0.5
        self.M_F = np.zeros(shape= (self.nb_tasks, self.len_mem, ), dtype = float) + 0.5
        self.index_update = [0] * self.nb_tasks

        # memory of cr and F in epoch
        self.epoch_M_cr:list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()

        # memory of delta fcost p and o in epoch
        self.epoch_M_w: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
    
    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        for skf in range(self.nb_tasks):
            new_cr = self.M_cr[skf][self.index_update[skf]]
            new_F = self.M_F[skf][self.index_update[skf]]

            new_index = (self.index_update[skf] + 1) % self.len_mem

            if len(self.epoch_M_cr) > 0:
                new_cr = np.sum(np.array(self.epoch_M_cr[skf]) * (np.array(self.epoch_M_w[skf]) / (np.sum(self.epoch_M_w[skf]) + 1e-50)))
                new_F = np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf]) ** 2) / \
                    (np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf])) + 1e-50)
            
            self.M_cr[skf][new_index] = new_cr
            self.M_F[skf][new_index] = new_F

            self.index_update[skf] = new_index

        # reset epoch mem
        self.epoch_M_cr:list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_w: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        
    def __call__(self, ind: Individual, population: Population, *args, **kwargs) -> Individual:
        super().__call__(*args, **kwargs)
        # random individual
        ind_ran1, ind_ran2 = population.__getIndsTask__(ind.skill_factor, size = 2, replace= False, type= 'random')


        if np.all(ind_ran1.genes == ind_ran2.genes):
            ind_ran2 = population[ind.skill_factor].__getWorstIndividual__

        # get best individual
        ind_best = population.__getIndsTask__(ind.skill_factor, type = self.p_best_type, p_ontop= self.p_ontop, tournament_size= self.tournament_size)
        while ind_best is ind:
            ind_best = population.__getIndsTask__(ind.skill_factor, type = self.p_best_type, p_ontop= self.p_ontop, tournament_size= self.tournament_size)


        k = np.random.choice(self.len_mem)
        cr = np.clip(np.random.normal(loc = self.M_cr[ind.skill_factor][k], scale = 0.1), 0, 1)

        F = 0
        while F <= 0 or F > 1:
            F = scipy.stats.cauchy.rvs(loc= self.M_F[ind.skill_factor][k], scale= 0.1) 
    
        u = (np.random.uniform(size = self.dim_uss) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (self.dim_uss,))
            u[np.random.choice(self.dim_uss)] = 1

        new_genes = np.where(u, 
            ind_best.genes + F * (ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        new_genes = np.clip(new_genes, 0, 1)

        new_ind = self.IndClass(new_genes)
        new_ind.skill_factor = ind.skill_factor
        new_ind.fcost = new_ind.eval(self.tasks[new_ind.skill_factor])

        # save memory
        delta = ind.fcost - new_ind.fcost
        if delta > 0:
            self.epoch_M_cr[ind.skill_factor].append(cr)
            self.epoch_M_F[ind.skill_factor].append(F)
            self.epoch_M_w[ind.skill_factor].append(delta)

        return new_ind


class L_SHADE(AbstractSearch):
    def __init__(self, len_mem = 30, p_best_type:str = 'ontop', p_ontop = 0.1, tournament_size = 2) -> None:
        '''
        `p_best_type`: `random` || `tournament` || `ontop`
        '''
        super().__init__()
        self.len_mem = len_mem
        self.p_best_type = p_best_type
        self.p_ontop = p_ontop
        self.tournament_size = tournament_size

    def getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed = None):
        super().getInforTasks(IndClass, tasks, seed= seed)
        # memory of cr and F
        self.M_cr = np.zeros(shape = (self.nb_tasks, self.len_mem, ), dtype= float) + 0.5
        self.M_F = np.zeros(shape= (self.nb_tasks, self.len_mem, ), dtype = float) + 0.5
        self.index_update = [0] * self.nb_tasks

        # memory of cr and F in epoch
        self.epoch_M_cr:list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()

        # memory of delta fcost p and o in epoch
        self.epoch_M_w: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
    
    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        for skf in range(self.nb_tasks):
            new_cr = self.M_cr[skf][self.index_update[skf]]
            new_F = self.M_F[skf][self.index_update[skf]]

            new_index = (self.index_update[skf] + 1) % self.len_mem

            if len(self.epoch_M_cr) > 0:
                new_cr = np.sum(np.array(self.epoch_M_cr[skf]) * (np.array(self.epoch_M_w[skf]) / (np.sum(self.epoch_M_w[skf]) + 1e-50)))
                new_F = np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf]) ** 2) / \
                    (np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf])) + 1e-50)
            
            self.M_cr[skf][new_index] = new_cr
            self.M_F[skf][new_index] = new_F

            self.index_update[skf] = new_index

        # reset epoch mem
        self.epoch_M_cr:list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_w: list[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        
    def __call__(self, ind: Individual, population: Population, *args, **kwargs) -> Individual:
        super().__call__(*args, **kwargs)
        # random individual
        ind_ran1, ind_ran2 = population.__getIndsTask__(ind.skill_factor, size = 2, replace= False, type= 'random')


        if np.all(ind_ran1.genes == ind_ran2.genes):
            ind_ran2 = population[ind.skill_factor].__getWorstIndividual__

        # get best individual
        ind_best = population.__getIndsTask__(ind.skill_factor, type = self.p_best_type, p_ontop= self.p_ontop, tournament_size= self.tournament_size)
        while ind_best is ind:
            ind_best = population.__getIndsTask__(ind.skill_factor, type = self.p_best_type, p_ontop= self.p_ontop, tournament_size= self.tournament_size)


        k = np.random.choice(self.len_mem)
        cr = np.clip(np.random.normal(loc = self.M_cr[ind.skill_factor][k], scale = 0.1), 0, 1)

        F = 0
        while F <= 0 or F > 1:
            F = scipy.stats.cauchy.rvs(loc= self.M_F[ind.skill_factor][k], scale= 0.1) 
    
        u = (np.random.uniform(size = self.dim_uss) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (self.dim_uss,))
            u[np.random.choice(self.dim_uss)] = 1

        new_genes = np.where(u, 
            ind.genes + F * (ind_best.genes - ind.genes + ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        new_genes = np.where(new_genes > 1, (ind.genes + 1)/2, new_genes) 
        new_genes = np.where(new_genes < 0, (ind.genes + 0)/2, new_genes) 

        new_ind = self.IndClass(new_genes)
        new_ind.skill_factor = ind.skill_factor
        new_ind.fcost = new_ind.eval(self.tasks[new_ind.skill_factor])

        # save memory
        delta = ind.fcost - new_ind.fcost
        if delta > 0:
            self.epoch_M_cr[ind.skill_factor].append(cr)
            self.epoch_M_F[ind.skill_factor].append(F)
            self.epoch_M_w[ind.skill_factor].append(delta)

        return new_ind

