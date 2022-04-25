from re import sub
import numpy as np

from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.function import AbstractFunc
from ..EA import *
import matplotlib.pyplot as plt


class model(AbstractModel.model): 
    def compile(self, tasks: list[AbstractFunc], crossover: Crossover.AbstractCrossover, mutation: Mutation.AbstractMutation, selection: Selection.AbstractSelection, *args, **kwargs):
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
        rmp = np.zeros((k,k))
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
    def renderRMP(self,tmp, title = None, figsize = None, dpi = 200):
        if figsize is None:
            figsize = (30, 30)
        if title is None:
            title = self.__class__.__name__
        fig = plt.figure(figsize= figsize, dpi = dpi)
        fig.suptitle(title, size = 15)
        fig.set_facecolor("white")
        for i in range(len(self.tasks)):
            for j in range(len(self.tasks)):
                x=[]
                if i !=j:
                    for k in range(1000):
                        x.append(tmp[k][i][j])
                    plt.subplot(int(np.ceil(len(self.tasks) / 3)), 3, i + 1)
                    plt.plot(x,label= 'task: ' +str(j + 1))
                    plt.legend()
              
            plt.title('task ' + str( i + 1))
            plt.xlabel("Epoch")
            plt.ylabel("M_rmp")
            plt.ylim(bottom = -0.1, top = 1.1)
    def RoutletWheel(self,rmp,rand):
        tmp = [0]*len(self.tasks)
        tmp[0]= rmp[0]
        for i in range(1,len(tmp)):
            tmp[i]=tmp[i-1]+rmp[i]
        index =0 
        while tmp[index] < rand:
            index+=1
            if index == 9:
                return index
        return index 
    def check(self,offspring,current_inds_each_task):
        for i in range(len(offspring.ls_subPop)):
            if len(offspring[i]) < current_inds_each_task[i]:
                return True
        return False
    def fit(self, max_inds_each_task: list, nb_generations :int ,LSA = False,  bound = [0, 1], evaluate_initial_skillFactor = False,
            *args, **kwargs): 
        super().fit(*args, **kwargs)

        current_inds_each_task = np.copy(max_inds_each_task) 


        population = Population(
            nb_inds_tasks= current_inds_each_task, 
            dim = self.dim_uss, 
            bound = bound, 
            list_tasks= self.tasks, 
            evaluate_initial_skillFactor= evaluate_initial_skillFactor
        )
        self.IM = []
        len_task  = len(self.tasks)
        for epoch in range(nb_generations):
            offspring = Population(
                nb_inds_tasks= [0] * len(self.tasks),
                dim = self.dim_uss, 
                bound = bound,
                list_tasks= self.tasks,
            )
            elite = self.get_elite(population.ls_subPop,20)
            rmp = self.get_pop_intersection_v2(elite)
            # create new offspring population 
            # while self.check(offspring,current_inds_each_task): 
            while len(offspring) <len(population):
                # choose parent 
                pa, pb = population.__getRandomInds__(size= 2) 
                # crossover
                if pa.skill_factor == pb.skill_factor or np.random.rand() < rmp[pa.skill_factor][pb.skill_factor]: 
                    oa, ob = self.crossover(pa, pb) 
                    oa.skill_factor, ob.skill_factor = np.random.choice([pa.skill_factor, pb.skill_factor], size= 2, replace= True)
                    if pa.skill_factor != pb.skill_factor:
                        oa.transfer =True
                        ob.transfer =True
                else: 
                    pa1 = self.findParentSameSkill(population[pa.skill_factor], pa) 
                    oa, _ = self.crossover(pa, pa1) 
                    #3.oa = self.mutation(pa, return_newInd= True)
                    oa.skill_factor = pa.skill_factor

                    pb1 = self.findParentSameSkill(population[pb.skill_factor], pb) 
                    ob, _ = self.crossover(pb, pb1) 
                    #ob = self.mutation(pb, return_newInd= True)
                    ob.skill_factor = pb.skill_factor 
                # eval and append # addIndividual already has eval  
                offspring.__addIndividual__(oa) 
                offspring.__addIndividual__(ob) 
            offspring.update_rank()
              
    
               # merge and update rank
            population = population + offspring 
            population.update_rank()
            # selection 
            self.selection(population, current_inds_each_task)


            # save history 
            self.history_cost.append([ind.fcost for ind in population.get_solves()])
            
            self.render_process((epoch + 1)/nb_generations, ["Cost"], [self.history_cost[-1]], use_sys= True)
            self.IM.append(np.copy(rmp))
        print("End")
        # solve 
        self.last_pop = population 
        return self.last_pop.get_solves()




