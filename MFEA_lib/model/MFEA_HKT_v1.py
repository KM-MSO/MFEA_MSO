from asyncio import tasks
from operator import itemgetter
from re import S, sub
import numpy as np
import operator

from sqlalchemy import false
from . import AbstractModel
from ..operators import Crossover, Mutation, Selection ,Search
from ..tasks.function import AbstractFunc
from ..EA import *
import matplotlib.pyplot as plt
import copy

class model(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: list[AbstractTask], 
        crossover: Crossover.SBX_Crossover, 
        mutation: Mutation.Polynomial_Mutation, 
        search: Search.SHADE,
        selection: Selection.ElitismSelection, 
        *args, **kwargs):
        super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)
        self.search = search
        self.search.getInforTasks(IndClass, tasks, seed = self.seed)
    
    def get_elite(self,pop,size):
        elite_subpops = []
        for i in range(len(pop)):
            idx = np.argsort(pop[i].factorial_rank)[:size].tolist()
            elite_subpops.append(pop[i][idx])
        return elite_subpops

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

    def fit(self, nb_inds_each_task: int, nb_inds_min:int,nb_generations :int ,  bound = [0, 1], evaluate_initial_skillFactor = False,LSA = False,
            *args, **kwargs): 
        super().fit(*args, **kwargs)
        population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )
        #history
        self.IM = []
        self.rmp_hist = []
        len_task  = len(self.tasks)
        
        #SA param
        nb_inds_tasks = [nb_inds_each_task]*len(self.tasks)
        MAXEVALS = nb_generations * nb_inds_each_task * len(self.tasks)
        epoch =0 
        eval_k = np.zeros(len(self.tasks))
        alpha =0.1

        rmp = np.zeros((len_task,len_task))
        while np.sum(eval_k) <= MAXEVALS:
            elite = self.get_elite(population.ls_subPop,size = 20)
            if epoch % 5 == 1 : 
                rmp = self.get_pop_intersection_v2(elite)
            if np.sum(eval_k) >= epoch * nb_inds_each_task * len(self.tasks):
                    # save history 
                self.history_cost.append([ind.fcost for ind in population.get_solves()])
                self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True)
                self.rmp_hist.append(np.copy(rmp))
                epoch+=1   
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks= [0] * len(self.tasks),
                dim =  self.dim_uss, 
                list_tasks= self.tasks,
            )
            # for k in range(len_task):
            #     if np.random.rand() > alpha: 
            #         for idx in range(nb_inds_tasks[k]):
            #             oa = self.search(ind = population[k][idx],population=population)
            #             offsprings.__addIndividual__(oa)
            #             eval_k[k]+=1
            #     else:
            #         # l = self.RoutletWheel(rmp[k],np.random.rand()) 
            #         l = 1-k
            #         while len(offsprings[k]) < nb_inds_tasks[k]:
            #             pa = population[k].__getRandomItems__()
            #             pb = population[l].__getRandomItems__()
            #             oa, ob = self.crossover(pa,pb,k,k)
            #             oa = self.mutation(oa, return_newInd= False)
            #             ob = self.mutation(ob, return_newInd= False)   
            #             offsprings.__addIndividual__(oa)
            #             offsprings.__addIndividual__(ob)
            #             eval_k[k] +=2
            # L_SHADE core 
            for k in range(len_task):
                for inv in range (nb_inds_tasks[k]):
                    t = np.random.randint(len_task)
                    if t == k :
                        oa = self.search(ind = population[k][inv],population=population)
                        offsprings.__addIndividual__(oa)
                        eval_k[k]+=1 
                    else:
                        if np.random.rand() < rmp[k][t]:
                            pa = population[k][inv]
                            pb = population[t].__getRandomItems__()
                            oa, ob = self.crossover(pa,pb,k,k)
                            # oa = self.mutation(oa, return_newInd= True)
                            # ob = self.mutation(ob, return_newInd= True)   
                            oa.fcost  = self.tasks[k](oa.genes)
                            ob.fcost  = self.tasks[k](ob.genes)
                            if oa.fcost < ob.fcost:
                                offsprings.__addIndividual__(oa)
                            else:
                                offsprings.__addIndividual__(ob)
                            eval_k[k] +=1
                        else:
                            oa = self.search(ind = population[k][inv],population=population)
                            offsprings.__addIndividual__(oa)
                            eval_k[k]+=1 
            # merge and update rank
            population = population + offsprings
            population.update_rank()

            # selection 
            if LSA is True:
                nb_inds_tasks = [int(
                    # (nb_inds_min - nb_inds_each_task) / nb_generations * (epoch - 1) + nb_inds_each_task
                    int(min((nb_inds_min - nb_inds_each_task)/(nb_generations - 1)* epoch  + nb_inds_each_task, nb_inds_each_task))
                )] * len(self.tasks)
            self.selection(population, nb_inds_tasks)
            if epoch > 0 :
                self.mutation.update(population)
            # self.search.update()
            # save history
            self.history_cost.append([ind.fcost for ind in population.get_solves()])
                
            self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True)

            # self.IM.append(np.copy(IM))
            self.rmp_hist.append(np.copy(rmp))
        print("End")
        # solve 
        self.last_pop = population 
        return self.last_pop.get_solves()




