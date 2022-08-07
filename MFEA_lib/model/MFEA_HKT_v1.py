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
        crossover: Crossover.SBX_Crossover, mutation: Mutation.GaussMutation, selection: Selection.ElitismSelection, 
        *args, **kwargs):
        super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)

    def cauchy_g(self, mu: float, gamma: float):
        return mu + gamma*np.tan(np.pi * np.random.rand()-0.5)

    def get_elite(self,pop,size):
        elite_subpops = []
        for i in range(len(pop)):
            idx = np.argsort(pop[i].factorial_rank)[:size].tolist()
            elite_subpops.append(pop[i][idx])
        return elite_subpops
    def get_elite_transfer (self, pop, size) :
        elite_transfer = []
        idx = np.argsort(pop.factorial_rank)[:size].tolist()
        elite_transfer += pop[idx]
        return elite_transfer
    def distance(self,a,b):
        return np.linalg.norm(a-b)        
    def rank_distance(self,subpops,x:Individual):
        dist = []
        for i in subpops:
            dist.append(self.distance(i.genes,x.genes))
        return np.argsort(np.argsort(dist)) + 1
    def get_pop_similarity(self,subpops):
        k = len(self.tasks)
        rmp = np.ones((k,k))
        for i in range(k):
            for j in range(k):
                x =  self.rank_distance(subpops[i],subpops[i][0])
                y = self.rank_distance(subpops[i],subpops[j][0])
                rmp[i][j] = np.sum([np.abs(i+1-x[i]) for i in range(len(x))]) / np.sum([np.abs(i+1-y[i]) for i in range(len(y))])
        return rmp
    def get_pop_intersection(self,subpops):
        k = len(self.tasks)
        rmp = np.zeros([k,k])
        for i in range(k):
            DT = 0           
            for u in range(20):
                tmp = 9999999
                for v in range(20):
                    if v != u: 
                        tmp =min(self.distance(subpops[i][u],subpops[i][v]),tmp)
                DT+=tmp
            for j in range(k):
                DA = 0 
                for u in range(20):
                    tmp = 9999999
                    for v in range(20):
                        tmp =min(self.distance(subpops[i][u],subpops[j][v]),tmp)
                    DA+=tmp
                if j != i:
                    rmp[i][j] = np.float64( DT / DA)
        return rmp
    def get_pop_intersection_v2(self,subpops):
        k = len(self.tasks)
        rmp = np.ones((k,k))
        for i in range(k):
            DT = 0        
            for u in range(20):
                DT+=self.distance(subpops[i][u],subpops[i][0])
            for j in range(k):
                DA = 0 
                for u in range(20):
                    DA+=self.distance(subpops[i][0],subpops[j][u])
                if j != i:
                    rmp[i][j] = np.float64( DT / DA)
        return rmp
    def RoutletWheel(self,rmp,rand):
        tmp = [0]*len(self.tasks)
        tmp[0]= rmp[0]
        for i in range(1,len(tmp)):
            tmp[i]=tmp[i-1]+rmp[i]
        index =0 
        while tmp[index] < rand:
            index+=1
            if index == len(self.tasks) - 1:
                return index
        return index 

    def distance_to_pop (self,inv, pop_elite, fRank) :
        sum = 0
        for individual in pop_elite :
            sum += self.distance(inv.genes, individual.genes)
        tmp = 1/sum * (1+1/fRank)
        return tmp
    def get_max_IM (self,IM_i) :
        b = {}
        for i in range(len(self.tasks)) :
            b[i] = IM_i[i]
        temp = sorted(b.items(), key = operator.itemgetter(1), reverse=True)
        return temp
    def CurrentToPBest(self,ind: Individual, pbest:Individual,r1:Individual,r2:Individual):
        rand_pos = np.random.randint(len(self.Cr))
        mu_cr = np.random.normal(loc = self.Cr[rand_pos],scale=0.1)
        mu_f = self.F[rand_pos] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
        j_rand = np.random.randint(len(ind.genes))
        child = np.zeros(len(ind.genes))
        for i in range(len(child)):
            if np.random.rand() <= mu_cr or i == j_rand:
                child[i] = ind.genes[i] + mu_f * ((pbest.genes[i]-ind.genes[i]) + r1.genes[i] -r2.genes[i])
            else:
                child[i] =ind.genes[i]
        child = np.clip(child,0,1)

        return self.IndClass(child)  



    def fit(self, nb_inds_each_task: int, nb_inds_min:int,nb_generations :int ,  bound = [0, 1], evaluate_initial_skillFactor = False,LSA = False,
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
                        offsprings.__addIndividual__(self.current_to_pbest(population[t], t, indiv))
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
                            offsprings.__addIndividual__(self.current_to_pbest(population[t], t, indiv))
                    eval_k[t]+=1
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
        self.last_pop = population
        return self.last_pop.get_solves()
