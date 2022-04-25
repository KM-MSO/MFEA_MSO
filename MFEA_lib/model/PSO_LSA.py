from copy import deepcopy
from os import popen
import numpy as np

from MFEA_lib.model.Gauss_Try import gauss_mutation

from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.task import AbstractTask
from ..EA import *
import matplotlib.pyplot as plt
import random

class Memory_SaMTPSO:
    def __init__(self, number_tasks, LP = 10) -> None:
        self.success_history = np.zeros(shape= (number_tasks, LP), dtype= float) 
        self.fail_history = np.zeros(shape= (number_tasks, LP), dtype= float) 

        self.next_position_update= 0 # index cột đang được update 
        self.isFocus = False 
        self.epsilon = 0.001 
        self.pb = 0.005 
        self.LP = LP 
        
        self.maintain = 10
        self.number_tasks = number_tasks 
    
    def update_history_memory(self, task_partner, success= True)->None: 
        if success: 
            self.success_history[task_partner][self.next_position_update] += 1 
        else: 
            self.fail_history[task_partner][self.next_position_update] += 1 
        
    def compute_prob(self)-> np.ndarray: 

        sum = np.clip(np.sum(self.success_history, axis= 1) + np.sum(self.fail_history, axis= 1), 0, 10000)
        sum_success = np.clip(np.sum(self.success_history, axis= 1), 0, 100000) 

        SRtk = sum_success / (sum + self.epsilon) + self.pb 
        p = SRtk / np.sum(SRtk) 
        #TODO: add focus  
        # 

        self.next_position_update = (self.next_position_update + 1) % self.LP 
        self.success_history[:, self.next_position_update] = 0 
        self.fail_history[:, self.next_position_update]  = 0 
        return p  

class model(AbstractModel.model): 
    def compile(self, tasks: list[AbstractTask], crossover: Crossover.AbstractCrossover, mutation: Mutation.AbstractMutation, selection: Selection.AbstractSelection, *args, **kwargs):
        return super().compile(tasks, crossover, mutation, selection, *args, **kwargs)
    
    def findParentSameSkill(self, subpop: SubPopulation, ind):
        ind2 = ind 
        while ind2 is ind: 
            ind2 = subpop.__getRandomItems__(size= 1)[0]
        
        return ind2 
    
    def Linear_population_size_reduction(self, evaluations, current_size_pop, max_eval_each_tasks, max_size, min_size):
        for task in range(len(self.tasks)):
            new_size = (min_size[task] - max_size[task]) * evaluations[task] / max_eval_each_tasks[task] + max_size[task] 

            new_size= int(new_size) 
            if new_size < current_size_pop[task]: 
                current_size_pop[task] = new_size 
        
        return current_size_pop 
    def fit(self, max_inds_each_task: list, min_inds_each_task: list, max_eval_each_task: list,LSA = False,  bound = [0, 1], evaluate_initial_skillFactor = False,
            log_oneline = False, num_epochs_printed = 20, *args, **kwargs): 
        super().fit(*args, **kwargs)
        current_inds_each_task = np.copy(max_inds_each_task) 
        eval_each_task = np.zeros_like(max_eval_each_task) 
        
        population = Population(
            nb_inds_tasks= current_inds_each_task, 
            dim = self.dim_uss, 
            bound = bound, 
            list_tasks= self.tasks, 
            evaluate_initial_skillFactor= evaluate_initial_skillFactor
        )

        p_matrix = np.ones(shape= (len(self.tasks), len(self.tasks)),dtype= float) / len(self.tasks) 

        memory_task = [Memory_SaMTPSO(len(self.tasks)) for i in range(len(self.tasks))]

        history_p_matrix= [] 
        epoch= 0 

        while np.sum(eval_each_task) < np.sum(max_eval_each_task): 
            offsprings = Population(
                    nb_inds_tasks= [0] * len(self.tasks),
                    dim = self.dim_uss, 
                    bound = bound,
                    list_tasks= self.tasks
                )
            
            task_partner = [[] for i in range(len(self.tasks))] 
            number_child_each_task = np.zeros(shape=(len(self.tasks)), dtype = int)

            index = deepcopy(current_inds_each_task) 
            index_child_each_tasks = [[] for i in range(len(self.tasks))] 
            while len(offsprings) < len(population): 

                for task in range(len(self.tasks)): 
                    while number_child_each_task[task] < current_inds_each_task[task]:
                        task2 = np.random.choice(np.arange(len(self.tasks)), p= p_matrix[task])
                        task2 = int(task2) 

                        if task == task2: 
                            # pa, pb = population[task].__getRandomItems__(size= 2, replace= True) 
                            # pa = population.__getIndsTask__(task, type= 'random') 
                            pa = population.__getIndsTask__(task, type= "tournament", tournament_size=2)
                            pb = population.__getIndsTask__(task, type= "tournament", tournament_size=2)
                            while pa is pb: 
                                pb = population.__getIndsTask__(task, type= "tournament", tournament_size=2)
                                
                        else: 
                            pa = population[task].__getRandomItems__(size= 1)[0] 
                            pb = population[task2].__getRandomItems__(size= 1)[0]
                        
                        #crossover 
                  
                        oa, ob = self.crossover(pa, pb) 
                        oa.skill_factor = ob.skill_factor = pa.skill_factor 
                        
                        #append adn eval 
                        offsprings.__addIndividual__(oa) 
                        offsprings.__addIndividual__(ob) 

                        # update task_partner 
                        task_partner[task].append(task2)
                        task_partner[task].append(task2)

                        index_child_each_tasks[task].append(index[task]) 
                        index_child_each_tasks[task].append(index[task] + 1) 

                        index[task] += 2
                        number_child_each_task[task] += 2 

                        eval_each_task[oa.skill_factor] += 1 
                        eval_each_task[ob.skill_factor] += 1 
            
            # linear size 
            if LSA is True: 
                current_inds_each_task = self.Linear_population_size_reduction(eval_each_task, current_inds_each_task, max_eval_each_task, max_inds_each_task, min_inds_each_task) 

            # merge
            population = population + offsprings 

            # selection 
            choose_ind = self.selection(population, current_inds_each_task)

            # update memory task 
            for task in range(len(self.tasks)):
                for idx in index_child_each_tasks[task] : 
                    
                    if idx in choose_ind[task]: 
                        memory_task[task].update_history_memory(task_partner[task][idx - index_child_each_tasks[task][0]], success= True) 
                    elif idx >= index_child_each_tasks[task][0]:
                        memory_task[task].update_history_memory(task_partner[task][idx - index_child_each_tasks[task][0]], success= False) 
                
                p = np.copy(memory_task[task].compute_prob())
                assert p_matrix[task].shape == p.shape 
                p_matrix[task] = p_matrix[task] * 0.9 + p * 0.1 

            if np.sum(eval_each_task) >= epoch * 100 * len(self.tasks):
            # save history
                self.history_cost.append([ind.fcost for ind in population.get_solves()])
                
                # save history rmp 
                history_p_matrix.append(np.copy(p_matrix))

                self.render_process(np.sum(eval_each_task)/ np.sum(max_eval_each_task),["cost", "eval"], [self.history_cost[-1], eval_each_task])

                epoch += 1 

        print("End")

        # solve 
        self.last_pop = population
        return self.last_pop.get_solves(), history_p_matrix 

