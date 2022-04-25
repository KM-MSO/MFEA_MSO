from copy import deepcopy
import numpy as np
import scipy.stats

from . import AbstractModel
from ..operators import Crossover, Mutation, Selection, Search
from ..tasks.task import AbstractTask
from ..EA import *
import sys
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

        self.next_position_update = (self.next_position_update + 1) % self.LP 
        self.success_history[:, self.next_position_update] = 0 
        self.fail_history[:, self.next_position_update]  = 0 
        return p  

class model(AbstractModel.model): 
    def compile(self, tasks: list[AbstractTask], 
        crossover: Crossover.AbstractCrossover, mutation: Mutation.NoMutation, selection: Selection.AbstractSelection, search: Search.AbstractSearch,
        *args, **kwargs):
        self.search = search
        self.search.getInforTasks(tasks, seed = self.seed)
        return super().compile(tasks, crossover, mutation, selection, *args, **kwargs)
    
    def findIndexIndividualSameSkillOntop(self, subpop: SubPopulation, ind: Individual = None, ontop: float = 1): 
        idx2 = -1
        list_ontop = np.where(subpop.scalar_fitness  >= 1/(len(subpop) * ontop))[0]
        assert len(list_ontop) > 0 
        while idx2 == -1:
            idx2 = int(np.random.choice(list_ontop, size = 1)) 
            if ind is not None and ind is subpop[idx2]: 
                idx2 = -1 
        
        return subpop[idx2] 


    def findParentSameSkill(self, subpop: SubPopulation, ind):
        ind2 = ind 
        while ind2 is ind : 
            ind2 = subpop.__getRandomItems__(size= 1)[0]
        
        return ind2 
    
    def Linear_population_size_reduction(self, evaluations, current_size_pop, max_eval_each_tasks, max_size, min_size):
        for task in range(len(self.tasks)):
            new_size = (min_size[task] - max_size[task]) * evaluations[task] / max_eval_each_tasks[task] + max_size[task] 

            new_size= int(new_size) 
            if new_size < min_size[task]: 
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

        # add Gauss 
        gauss = [gauss_mutation(population[i]) for i in range(len(self.tasks))]
        
        # add DE 
        ti_le_DE_gauss=np.zeros(shape= (len(self.tasks)), dtype= float) + 0.5


        time_for_adapt = 0 
        epoch = 0 
        de_thanh_cong = [] 
        sbx_thanh_cong = [] 
        sbx_khong_thanh_cong = [] 
        while np.sum(eval_each_task) <= np.sum(max_eval_each_task): 
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
            # while len(offsprings) < len(population): 
            time_for_adapt += 1  
            
            for task in range(len(self.tasks)):
                count = 0 
                while number_child_each_task[task] < current_inds_each_task[task]:
                    task2 = np.random.choice(np.arange(len(self.tasks)), p= p_matrix[task])
                    task2 = int(task2) 

                    if task == task2: 
                        # pa, pb = population[task].__getRandomItems__(size= 2, replace= False)
                        # pa = population[task].__getRandomItems__(size= 1)[0] 
                        # pb = pa 
                        # while pa is pb and np.any(): 
                        #     pb1, pb2 = population[task].__getRandomItems__(size= 2) 
                        #     if pb1.fcost < pb2.fcost: 
                        #         pb = pb1 
                        #     else: 
                        #         pb = pb2 
                        # pa, pb = population[task].__getRandomItems__(size= 2, replace= False) 
                        # pa = population.__getIndsTask__(task, type= "random")
                        # pb = population.__getIndsTask__(task, type= 'tournament', tournament_size= 2) 
                        # while pa is pb : 
                        #     pb = population.__getIndsTask__(task, type= 'tournament', tournament_size= 2) 
                        pa, pb = population[task].__getRandomItems__(size= 2, replace=False)

                        
                    else: 
                        pa = population[task].__getRandomItems__(size= 1)[0] 
                        pb = population[task2].__getRandomItems__(size= 1)[0]

                    # pa = population[task][count]
                    # count += 1 
                    # if task == task2: 
                    #     pb = self.findParentSameSkill(population[task], pa) 
                    # else: 
                    #     pb = population[task2].__getRandomItems__(size= 1)[0] 
                    
                    
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

            # # merge
            population = population + offsprings 

            # # selection 
            # # self.selection.ontop -= 0.4 * np.sum(eval_each_task) / np.sum(max_eval_each_task)
            choose_ind = self.selection(population, current_inds_each_task)
            # self.selection.ontop = 0.9 
            # update memory task 
            for task in range(len(self.tasks)):
                count = 0 
                for idx in index_child_each_tasks[task] : 
                    
                    if idx in choose_ind[task]: 
                        count += 1 
                        memory_task[task].update_history_memory(task_partner[task][idx - index_child_each_tasks[task][0]], success= True) 
                    elif idx >= index_child_each_tasks[task][0]:
                        memory_task[task].update_history_memory(task_partner[task][idx - index_child_each_tasks[task][0]], success= False) 
                if task == 0 : 
                    sbx_thanh_cong.append(count)
                    sbx_khong_thanh_cong.append(len(index_child_each_tasks[task]) - count)
                p = np.copy(memory_task[task].compute_prob())
                assert p_matrix[task].shape == p.shape 
                p_matrix[task] = p_matrix[task] * 0.8 + p * 0.2 
            
            # save history
            
            

            # diver = [] 
            # is_jam = [] 
            # # # update gauss 
            # for task in range(len(self.tasks)):
            #     gauss[task].update_scale(population[task].getSolveInd().fcost) 
            #     is_jam.append(gauss[task].is_jam)

            # for subpop in population: 
            #     for idx in range(len(subpop)):
            #         if np.random.rand() < 0.5: 
            #             if np.random.rand() < 1.1: 
            #                 '''
            #                 DE
            #                 '''
            #                 new_ind = self.search(ind = subpop[idx], population= population)
            #                 if new_ind.fcost < subpop[idx].fcost: 
            #                     subpop.ls_inds[idx] = new_ind
            #                     # subpop.update_rank()
            #                 eval_each_task[subpop.skill_factor] += 1 
            # self.search.update() 
                            

            

            # # apply gauss mutation 
            # if time_for_adapt > -1: 
            #     time_for_adapt = 0 # sau khi dot bien va DE cho time de con tinh toan :) 
            #     for task in range(len(self.tasks)):

            #         count_mutation = 0 
            #         count_de = 0 
            #         eval_de =0 
            #         eval_gauss = 0 

            #         idx_mutation = np.random.choice(np.arange(current_inds_each_task[task]),size= int(current_inds_each_task[task]/2), replace= False)

            #         for idx in idx_mutation: 

            #             p = deepcopy(population[task][idx])
            #             #FIXME 
            #             # if np.random.uniform() < ti_le_DE_gauss[task] and gauss[task].is_jam is False: 
            #             if np.random.uniform() < 1.1 :
                        
            #                 # pbest = self.findIndexIndividualSameSkillOntop(population[task], ontop = 0.1)  
            #                 # p1 = pbest 
            #                 # p2 = pbest 
            #                 # while p1 is pbest or p2 is pbest or p1 is p2:  
            #                 #     # p1, p2 = population[task].__getRandomItems__(size = 2, replace= False)
            #                 #     p1 = population[task][int(np.random.choice(np.where(population[task].scalar_fitness >= 1/(self.selection.ontop * len(population[task])))[0], size= 1))]
            #                 #     p2 = population[task][int(np.random.choice(np.where(population[task].scalar_fitness >= 1/(self.selection.ontop * len(population[task])))[0], size= 1))]
            #                 #     # p2 = population[task][int(np.random.choice(population[task].scalar_fitness >= 1/(self.selection.ontop * len(population[task])), size= 1))]
            #                 #     pass
            #                 idx_pbest = idx_p1 = idx_p2 = idx 
            #                 while idx_pbest == idx or idx_p1 == idx_p2 or idx_p2 == idx_pbest: 
            #                     idx_pbest = int(np.random.choice(np.where(population[task].scalar_fitness >= 1/ (0.1 * current_inds_each_task[task]))[0], size= 1))
            #                     idx_p1 = int(np.random.choice(np.where(population[task].scalar_fitness >= 1/ (self.selection.ontop * current_inds_each_task[task]))[0], size= 1))
            #                     idx_p2 = int(np.random.choice(np.where(population[task].scalar_fitness >= 1/ (self.selection.ontop * current_inds_each_task[task]))[0], size= 1))
                            
            #                 pbest = population[task][idx_pbest] 
            #                 p1 = population[task][idx_p1]
            #                 p2 = population[task][idx_p2]

            #                 new_p = de[task].DE_cross(p.genes, pbest.genes, p1.genes, p2.genes) 
            #                 new_fcost = self.tasks[task].func(new_p)
            #                 eval_each_task[task] += 1 
            #                 eval_de += 1 
            #                 if new_fcost > 0: 
            #                     #NOTE
            #                     prob = population[task].getSolveInd().fcost / new_fcost 
                            
            #                 # if new_fcost <= p.fcost or (np.random.rand() < prob and population[task].factorial_rank[idx] >=  current_inds_each_task[task]/2): 
            #                 if new_fcost < p.fcost: 
            #                     if new_fcost < p.fcost : 
            #                         count_de += 1 
            #                         de[task].update(p.fcost - new_fcost) 
            #                     population[task][idx].genes = np.copy(new_p)
            #                     population[task][idx].fcost = new_fcost 
                        
            #             else: 
            #                 # new_p, std = gauss[task].mutation(p, population[task])
            #                 # assert type(new_p) == Individual, 'hmmm mutation ra ko giong :)) '
            #                 # new_p.fcost= new_p.eval(self.tasks[task]) 
            #                 # eval_each_task[task] += 1 
            #                 # eval_gauss += 1 
            #                 # if new_p.fcost > 0: 
            #                 #     prob = np.exp(-np.power(gauss[task].tiso, 1/2)) * (population[task].getSolveInd().fcost/ new_p.fcost)**2 * 0.5 
            #                 #     prob = population[task].getSolveInd().fcost / new_p.fcost
            #                 # # if new_p.fcost > 0: 
            #                 #     # prob = np.exp(-np.power(std, 1/2)) * (population[task].getSolveInd().fcost/ new_p.fcost)**2 * 0.5
            #                 # if new_p.fcost <= p.fcost or (np.random.rand() < prob and population[task].factorial_rank[idx] >=  current_inds_each_task[task]/2): 
            #                 #     if new_p.fcost < p.fcost :
            #                 #         count_mutation += 1 
            #                 #         pass  
            #                 #     population[task][idx].genes = new_p.genes 
            #                 #     population[task][idx].fcost = new_p.fcost 
            #                 pass

            #         if task == 0: 
            #             de_thanh_cong.append(count_de) 
            #         if eval_de > 0 and eval_gauss > 0: 
            #             a = count_de / eval_de
            #             b= count_mutation / eval_gauss
            #             if a== b and a == 0: 
            #                 ti_le_DE_gauss[task] -= (ti_le_DE_gauss[task] - 0.5) * 0.2
            #             else: 
            #                 x= np.max([a / (a + b), 0.1]) 
            #                 x= np.min([x, 0.9]) 
            #                 ti_le_DE_gauss[task] = ti_le_DE_gauss[task]* 0.5+ x * 0.5  
                    
            #     # update de 
            #     for d in de: 
            #         d.reset() 

            # render progress 
            if np.sum(eval_each_task) >= epoch * 100 * len(self.tasks):
                # save history
                self.history_cost.append([ind.fcost for ind in population.get_solves()])
                history_p_matrix.append(np.copy(p_matrix))
                #print
                # self.render_process(np.sum(eval_each_task)/ np.sum(max_eval_each_task), ["cost", "eval", "ti_le_gauss"], [self.history_cost[-1], eval_each_task, ti_le_DE_gauss.tolist()], use_sys= True)
                self.render_process(np.sum(eval_each_task)/ np.sum(max_eval_each_task), ["cost", "eval", "ti_le_gauss"], [self.history_cost[-1], eval_each_task, ti_le_DE_gauss.tolist()], use_sys= True)

                epoch = max(int(np.sum(eval_each_task) / (100 * len(self.tasks))), epoch + 1) 
            
            # self.render_process_terminal(np.sum(eval_each_task)/ np.sum(max_eval_each_task), ["cost", "is_jam", "ti_le_gauss"], [self.history_cost[-1], is_jam, ti_le_DE_gauss.tolist()])
            
        print("End")
        print(epoch)

        # solve 
        self.last_pop = population
        self.result = self.history_cost[-1]
        return self.last_pop.get_solves(), history_p_matrix , de_thanh_cong, sbx_thanh_cong, sbx_khong_thanh_cong

                     

                        


class gauss_mutation:
    def __init__(self, subpopu : SubPopulation) -> None:
        self.is_jam = False
        self.scale = None

        self.cost_10_pre_gen = None
        self.curr_cost = None
        self.count_gen = 0

        self.max_log = np.zeros(shape=(subpopu.dim, ), dtype=int) + 2
        self.min_log = 1

        self.count_mu_each_dim = np.zeros(shape=(subpopu.dim,), dtype=int)
        self.diversity_begin: float = -1
        self.curr_diversity: float = -1

        self.count_mu_each_scale = np.zeros(shape=(subpopu.dim, 30), dtype=int)
        self.his_scale_each_dim = [[] for i in range(subpopu.dim,)]



        self.D0 = 0
        self.tiso = 1

        self.e_tmp = -1 
        self.i_tmp = -1 

        if subpopu.getSolveInd().fcost > 0:
            sum_cost = np.sum(np.array([i.fcost for i in subpopu.ls_inds]))
            for ind in subpopu.ls_inds:
                w = 1 - ind.fcost / sum_cost
                self.D0 += w * \
                    (np.sqrt(
                        np.sum((np.array((ind - subpopu.getSolveInd()).genes)) ** 2)))


    def update_scale_base_population(self, subpopu: SubPopulation):
        Dt = 0
        a = -1 
        if  subpopu.getSolveInd().fcost > 0 :
            sum_cost = np.sum(np.array([i.fcost for i in subpopu.ls_inds]))
            for ind in subpopu.ls_inds:
                w = 1 - ind.fcost / sum_cost
                Dt += w * \
                    (np.sqrt(
                        np.sum((np.array((ind - subpopu.getSolveInd()).genes)) ** 2)))


            a = self.D0 / (self.D0 - Dt)
        # self.tiso = 1/a 
        self.tiso = Dt 

    def update_scale(self, best_cost):
        if self.cost_10_pre_gen is None and self.curr_cost is None:
            self.curr_cost = best_cost
            self.cost_10_pre_gen = best_cost
        else:
            self.curr_cost = best_cost
            self.count_gen += 1
        if self.count_gen > 10 and self.cost_10_pre_gen > 0:
            delta = (self.cost_10_pre_gen - self.curr_cost) / \
                self.cost_10_pre_gen
            if delta < 0.1:
                self.is_jam = True
            else:
                self.is_jam = False

            self.cost_10_pre_gen = best_cost
            self.count_gen = 0

    def mutation(self, individual: Individual, subpopulation: SubPopulation, jam_and_mu = False)->Individual:
        D = subpopulation.dim
        ind = deepcopy(individual)
        p_for_dim = 1 / (self.count_mu_each_dim + 1)
        p_for_dim /= np.sum(p_for_dim)
        i = int(np.random.choice(np.arange(D), size=1, p=p_for_dim))
        # mean = np.mean(subpopulation[:, i])
        std = np.std([subpopulation[j].genes[i] for j in range(len(subpopulation))])
        e = -1

        if std != 0:
            log = np.log10(std)
            log = int(np.abs(log))+1
            if log+1 > self.max_log[i]:
                self.max_log[i] = log + 1
                # NOTE
                self.count_mu_each_dim = np.zeros_like(self.count_mu_each_dim) + 1
                self.count_mu_each_scale[i] = np.zeros_like(
                    self.count_mu_each_scale[i]) + 1

        rand_scale = 1 / (self.count_mu_each_scale[i][1:self.max_log[i]] + 1)

        p = np.array(rand_scale / np.sum(rand_scale))
        # if rand <= 0.1 and self.is_jam is False:
        # if self.is_jam is False: 
        # if False:
        if jam_and_mu:
            # e = int(self.max_log[i]) - 1 
            # assert e > 0 
            # e = -e 
            # self.scale = 10 ** int(e)
            return ind, std
        else: 
            e = int(np.random.choice(
                np.arange(self.max_log[i])[1:], size=1, p=p))
            e = -e
            self.scale = 10 ** e
            self.count_mu_each_dim[i] += 1
            self.count_mu_each_scale[i][np.abs(e)] += 1
            self.his_scale_each_dim[i].append(np.abs(e))
        t = ind[i] + np.random.normal(loc=0, scale=np.abs(self.scale))

        #save e and i 
        self.e_tmp = np.abs(e)  
        self.i_tmp = i 
        
        if t > 1:
            t = ind[i] + np.random.rand() * (1 - ind[i])
        elif t < 0:
            t = np.random.rand() * np.abs(ind[i])

        ind.genes[i] = t 

        return ind, std 
    
    def success(self): 
        self.count_mu_each_scale[self.i_tmp][self.e_tmp] -= 1 
        


