import numpy as np


from . import AbstractModel
from ..operators import Crossover, Mutation, Selection ,Search
from ..tasks.function import AbstractFunc
from ..EA import *
import copy

  
class model(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: list[AbstractTask], 
        crossover: Crossover.SBX_Crossover, 
        mutation: Mutation.Polynomial_Mutation, 
        selection: Selection.ElitismSelection, 
        *args, **kwargs):
        super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)

    def cauchy_g(self, mu: float, gamma: float):
        return mu + gamma*np.tan(np.pi * np.random.rand()-0.5)

    def get_elite(self,sub_pop,size):
        elite_subpops = []
        idx_elites = np.argsort(sub_pop.factorial_rank)[:size]
        for idx in idx_elites :
            elite_subpops.append(sub_pop[idx])
        return elite_subpops   

    def current_to_pbest(self, sub_pop: SubPopulation, id_task: int , curr_indiv: Individual) -> Individual:
        rand_pos = np.random.randint(self.H)
        mu_cr = self.mem_cr[id_task,rand_pos]
        mu_f = self.mem_f[id_task, rand_pos]

        if (mu_cr == -1):
            cr = 0
        else:
            cr = np.clip(np.random.normal(loc=mu_cr, scale=0.1), 0, 1)
            
        while True:
            f = self.cauchy_g(mu_f, gamma=0.1)
            if f > 0:
                break
        if f > 1:
            f = 1
        
        
        pbest_size = int(self.BEST_RATE * len(sub_pop))
        idx_elites = np.argsort(sub_pop.factorial_rank)[:pbest_size]
        
        # pbest = curr_indiv
        # while pbest == curr_indiv:
        pbest = sub_pop[np.random.choice(idx_elites)]

        r1 = curr_indiv
        # while r1 == curr_indiv or r1 == pbest:
        r1 = sub_pop.__getRandomItems__()
        if self.update_time[id_task] > 0 and np.random.rand() <= len(self.archive[id_task]) / (len(self.archive[id_task]) + len(sub_pop)):
            r2 = self.archive[id_task][np.random.randint(len(self.archive[id_task]))]
        else:
            r2 = curr_indiv
            # while r2 == curr_indiv or r2 == r1 or r2 == pbest:
            r2 = sub_pop.__getRandomItems__()
        
        j_rand = np.random.randint(len(curr_indiv))
        temp_genes = np.random.rand(len(curr_indiv))
        for j in range(len(curr_indiv)):
            if np.random.rand() <= cr or j == j_rand:
                temp_genes[j] = curr_indiv[j] + f * (pbest[j] - curr_indiv[j] + r1[j] - r2[j])
                # bound handling
                if temp_genes[j] > 1:
                    temp_genes[j] = (curr_indiv[j] + 1)/2
                elif temp_genes[j] < 0:
                    temp_genes[j] = (curr_indiv[j] + 0)/2
            else:
                temp_genes[j] = curr_indiv[j]
        child = self.IndClass(temp_genes)
        child.fcost = sub_pop.task(child.genes)
        child.skill_factor = id_task
        self.count_evals += 1
        
        if child.fcost == curr_indiv.fcost:
            return child
        elif child.fcost < curr_indiv.fcost:
            self.success_cr[id_task].append(cr)
            self.success_f[id_task].append(f)
            self.diff_fitness[id_task].append(curr_indiv.fcost - child.fcost)
            if len(self.archive[id_task]) < self.ARC_RATE * len(sub_pop):
                self.archive[id_task].append(curr_indiv)
            else:
                self.archive[id_task].pop(np.random.randint(len(self.archive[id_task])))
                self.archive[id_task].append(curr_indiv)
            return child
        else:
            return curr_indiv

    def rand_1(self, sub_pop: SubPopulation, id_task: int , curr_indiv: Individual) -> Individual:
        rand_pos = np.random.randint(self.H)
        mu_cr = self.mem_cr[id_task,rand_pos]
        mu_f = self.mem_f[id_task, rand_pos]

        if (mu_cr == -1):
            cr = 0
        else:
            cr = np.clip(np.random.normal(loc=mu_cr, scale=0.1), 0, 1)
            
        while True:
            f = self.cauchy_g(mu_f, gamma=0.1)
            if f > 0:
                break
        if f > 1:
            f = 1

        r1 = curr_indiv
        # while r1 == curr_indiv or r1 == pbest:
        r1 = sub_pop.__getRandomItems__()
        r3 = sub_pop.__getRandomItems__()
        if self.update_time[id_task] > 0 and np.random.rand() <= len(self.archive[id_task]) / (len(self.archive[id_task]) + len(sub_pop)):
            r2 = self.archive[id_task][np.random.randint(len(self.archive[id_task]))]
        else:
            r2 = curr_indiv
            # while r2 == curr_indiv or r2 == r1 or r2 == pbest:
            r2 = sub_pop.__getRandomItems__()
        
        j_rand = np.random.randint(len(curr_indiv))
        temp_genes = np.random.rand(len(curr_indiv))
        for j in range(len(curr_indiv)):
            if np.random.rand() <= cr or j == j_rand:
                temp_genes[j] = r1[j] + f * (r3[j] - r2[j])
                # bound handling
                if temp_genes[j] > 1:
                    temp_genes[j] = (curr_indiv[j] + 1)/2
                elif temp_genes[j] < 0:
                    temp_genes[j] = (curr_indiv[j] + 0)/2
            else:
                temp_genes[j] = curr_indiv[j]
        child = self.IndClass(temp_genes)
        child.fcost = sub_pop.task(child.genes)
        child.skill_factor = id_task
        self.count_evals += 1
        
        if child.fcost == curr_indiv.fcost:
            return child
        elif child.fcost < curr_indiv.fcost:
            self.success_cr[id_task].append(cr)
            self.success_f[id_task].append(f)
            self.diff_fitness[id_task].append(curr_indiv.fcost - child.fcost)
            if len(self.archive[id_task]) < self.ARC_RATE * len(sub_pop):
                self.archive[id_task].append(curr_indiv)
            else:
                self.archive[id_task].pop(np.random.randint(len(self.archive[id_task])))
                self.archive[id_task].append(curr_indiv)
            return child
        else:
            return curr_indiv
    def Uniform_Crossover(self, pa: Individual, pb: Individual, skf_oa= None) -> Individual:
        genes = np.zeros(self.dim_uss)
        for i in range(self.dim_uss) :
            if np.random.random() < 0.5 :
                genes[i] = pa.genes[i]
            else : 
                genes[i] = pb.genes[i]
        oa = self.IndClass(genes)
        oa.skill_factor = skf_oa
        return oa
    def update_state(self, sub_pop: SubPopulation, id_task: int):
        self.update_time[id_task] += 1

        # Update F, CR memory
        if len(self.success_cr[id_task]) > 0:
            self.mem_cr[id_task][self.mem_pos[id_task]] = 0
            self.mem_f[id_task][self.mem_pos[id_task]] = 0
            temp_sum_cr = 0
            temp_sum_f = 0
            sum_diff = 0

            for d in self.diff_fitness[id_task]:
                sum_diff += d
            for i in range(len(self.success_cr[id_task])):
                weight = self.diff_fitness[id_task][i] / sum_diff
                
                self.mem_f[id_task][self.mem_pos[id_task]] += weight * self.success_f[id_task][i] ** 2
                temp_sum_f += weight * self.success_f[id_task][i]
                
                self.mem_cr[id_task][self.mem_pos[id_task]] += weight * self.success_cr[id_task][i] **2
                temp_sum_cr += weight * self.success_cr[id_task][i]
            
            self.mem_f[id_task][self.mem_pos[id_task]] /= temp_sum_f

            if temp_sum_cr == 0 or self.mem_cr[id_task][self.mem_pos[id_task]] == -1:
                self.mem_cr[id_task][self.mem_pos[id_task]] = -1
            else:
                self.mem_cr[id_task][self.mem_pos[id_task]] /= temp_sum_cr
            
            self.mem_pos[id_task] += 1

            if self.mem_pos[id_task] >= self.H:
                self.mem_pos[id_task] = 0
            self.success_cr[id_task].clear()
            self.success_f[id_task].clear() 
            self.diff_fitness[id_task].clear()

        # Update rank
        sub_pop.update_rank()
        
        # Update archive
        while len(self.archive[id_task]) > self.ARC_RATE * len(sub_pop):
            self.archive[id_task].pop(np.random.randint(len(self.archive[id_task])))

        # Update RMP
            self.best_partner[id_task] = -1
            max_rmp = 0
            for other_task in range(len(self.tasks)):
                if other_task != id_task:
                    good_mean = 0
                    if len(self.success_rmp[(id_task, other_task)]) > 0:
                        sum = 0
                        for d in self.diff_f_inter_x[(id_task, other_task)]:
                            sum += d
                        val1, val2 = 0, 0
                        for k in range(len(self.success_rmp[(id_task, other_task)])):
                            w = self.diff_f_inter_x[(id_task, other_task)][k] / sum
                            val1 += w * self.success_rmp[(id_task, other_task)][k] ** 2
                            val2 += w * self.success_rmp[(id_task, other_task)][k] 
                        good_mean = val1 / val2

                        if good_mean > self.rmp[id_task][other_task] and good_mean > max_rmp:
                            max_rmp = good_mean
                            self.best_partner[id_task] = other_task
                    
                    if good_mean > 0:
                        c1 = 1
                    else:
                        c1 = 1 - self.C
                    self.rmp[id_task][other_task] = c1 * self.rmp[id_task][other_task] + self.C * good_mean
                    self.rmp[id_task][other_task] = np.max((0.01, np.min((1, self.rmp[id_task][other_task]))))

                    self.success_rmp[(id_task, other_task)].clear()
                    self.diff_f_inter_x[(id_task, other_task)].clear()
        
    def fit(self, nb_inds_each_task: int, nb_generations :int ,  nb_inds_min:int, evaluate_initial_skillFactor = False,LSA = False,
            *args, **kwargs): 
        super().fit(*args, **kwargs)
        
        # Const
        MAX_EVALS_PER_TASK = 100000
        EPSILON = 5e-7
        self.ARC_RATE = 5
        self.BEST_RATE = 0.11
        self.H = 30
        self.C = 0.02
        INIT_RMP = 0.5
        self.J = 0.3

        # Initialize the parameter
        num_tasks = len(self.tasks)
        
        self.mem_cr = np.full((num_tasks, self.H), 0.5, dtype=np.float64)
        self.mem_f = np.full((num_tasks, self.H), 0.5)
        self.update_time = np.full((num_tasks, ), 0, dtype=np.int64)
        self.mem_pos = np.full((num_tasks, ), 0, dtype=np.int64)
        self.count_evals = 0
        
        self.best_partner = np.full((num_tasks, ), -1, dtype=np.int64)
        self.update_time = np.full((num_tasks, ), 0, dtype=np.int64)
        self.rmp = np.full((num_tasks, num_tasks), INIT_RMP)
        self.success_rmp = {}
        self.diff_f_inter_x = {}
        self.archive = []
        self.success_cr = []
        self.success_f = []
        self.diff_fitness = []
        for task in range(num_tasks):
            self.archive.append([])
            self.success_cr.append([])
            self.success_f.append([])
            self.diff_fitness.append([])
            for other_task in range(num_tasks):
                self.success_rmp[(task, other_task)] = []
                self.diff_f_inter_x[(task, other_task)] = []

        # Initialize the population
        population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * num_tasks, 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )
        epoch = 1
        eval_k = np.zeros(len(self.tasks))
        nb_inds_tasks = [nb_inds_each_task] * len(self.tasks)
        # stop = False
        # while (self.count_evals < self.max_evals and (not stop)):
        while np.sum(eval_k) <= MAX_EVALS_PER_TASK*len(self.tasks):
            # stop = True
            for t in range(num_tasks):
                if population[t].__getBestIndividual__.fcost < EPSILON:
                    continue
                offsprings = SubPopulation(
                    IndClass=self.IndClass,
                    skill_factor=t,
                    dim=self.dim_uss,
                    num_inds=0,
                    task=self.tasks[t]
                )
                for indiv in population[t]:
                    other_t = np.random.randint(num_tasks)
                    if other_t == t:
                        if random.random() < 0.9 :
                            offsprings.__addIndividual__(self.current_to_pbest(population[t], t, indiv))
                        else :
                            offsprings.__addIndividual__(self.rand_1(population[t], t, indiv))
                    else:
                        if self.best_partner[t] == other_t:
                            rmp = 1
                        else:
                            mu_rmp = self.rmp[t, other_t]
                            while True:
                                rmp = np.random.normal(loc=mu_rmp, scale=0.1)
                                if not(rmp <= 0 or rmp > 1):
                                    break
                        if np.random.rand() <= rmp:
                            # Inter-task crossover
                            eval_k[t]+=1
                            other_indiv = population[other_task].__getRandomItems__()
                            oa, ob = self.crossover(indiv, other_indiv, t, t)
                            oa.fcost  = self.tasks[t](oa.genes)
                            ob.fcost  = self.tasks[t](ob.genes)

                            # Select better individual from 2 offsprings
                            survival = oa
                            if survival.fcost > ob.fcost:
                                survival = ob
                            
                            delta_fitness = indiv.fcost - survival.fcost
                            if (delta_fitness == 0):
                                offsprings.__addIndividual__(survival)
                            elif delta_fitness > 0:
                                self.success_rmp[(t, other_t)].append(rmp)
                                self.diff_f_inter_x[(t, other_t)].append(delta_fitness)
                                offsprings.__addIndividual__(survival)
                            else:
                                offsprings.__addIndividual__(indiv) 
                        else:
                            # Intra - crossover
                            if random.random() < 0.9 :
                                offsprings.__addIndividual__(self.current_to_pbest(population[t], t, indiv))
                            else :
                                offsprings.__addIndividual__(self.rand_1(population[t], t, indiv))
                    eval_k[t]+=1
                if epoch > 500 :
                    top10 = self.get_elite(population[t],10)
                    random.shuffle(top10)
                    if epoch %2 == 0 :
                        for j in range(5) :
                            oa,ob = self.crossover(top10[j], top10[j+5], t, t)
                            oa.fcost  = self.tasks[t](oa.genes)
                            ob.fcost  = self.tasks[t](ob.genes)
                            offsprings.__addIndividual__(oa)
                            offsprings.__addIndividual__(ob)
                            eval_k[t]+=2
                    else :
                        subpop_elite = np.zeros([10, self.dim_uss])
                        for j in range(10) :
                            subpop_elite[j] = top10[j].genes
                        mean = np.mean(subpop_elite, axis = 0)
                        std = np.std(subpop_elite, axis = 0)
                        for j in range(10) :
                            genes = np.zeros(self.dim_uss)
                            for l in range(self.dim_uss) :
                                genes[l] = np.random.normal(loc = mean[l], scale = std[l])
                            indiv = self.IndClass(genes = genes)
                            indiv.skill_factor = t
                            indiv.fcost = self.tasks[t](genes)
                            offsprings.__addIndividual__(indiv)
                            eval_k[t] += 1
                if np.random.rand() < self.J:
                    a = np.amax(offsprings.ls_inds,axis= 0)
                    b = np.amin(offsprings.ls_inds,axis= 0)
                    op_offsprings = SubPopulation(
                        IndClass=self.IndClass,
                        skill_factor=t,
                        dim=self.dim_uss,
                        num_inds=0,
                        task=self.tasks[t]
                    )
                    for inv in offsprings:
                        oa = self.IndClass(a+b-inv.genes)
                        oa.fcost = self.tasks[t](oa.genes)
                        op_offsprings.__addIndividual__(oa)
                        eval_k[t]+=1
                    offsprings  = offsprings+op_offsprings 
                # offsprings.update_rank()
                # self.selection(offsprings,nb_inds_tasks[t])
                population.ls_subPop[t] = offsprings  

                # Update RMP, F, CR, population size
                self.update_state(population[t], t)
            if LSA is True: 
                nb_inds_tasks = [int(
                    # (nb_inds_min - nb_inds_each_task) / nb_generations * (epoch - 1) + nb_inds_each_task
                    int(min((nb_inds_min - nb_inds_each_task)/(nb_generations - 1)* (epoch - 1) + nb_inds_each_task, nb_inds_each_task))
                )] * len(self.tasks)

            population.update_rank()
            self.selection(population, nb_inds_tasks)
            self.history_cost.append([indiv.fcost for indiv in population.get_solves()])
            if np.sum(eval_k) >= epoch * nb_inds_each_task * len(self.tasks):
                # save history
                self.history_cost.append([ind.fcost for ind in population.get_solves()])
                self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True)
                epoch +=1
            if np.sum(eval_k) >= nb_generations * nb_inds_each_task * len(self.tasks):
                epoch = nb_generations
                # save history
                self.history_cost.append([ind.fcost for ind in population.get_solves()])
                self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True)
                epoch +=1
        self.last_pop = population
        return self.last_pop.get_solves()
