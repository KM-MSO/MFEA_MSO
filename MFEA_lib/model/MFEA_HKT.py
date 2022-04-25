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

    def fit(self, max_inds_each_task: list, nb_generations :int ,  bound = [0, 1], evaluate_initial_skillFactor = False,
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
        self.rmp_hist = []
        len_task  = len(self.tasks)
        inter =  [0.9]*len_task
        intra =  [0.1]*len_task
        rmp = np.zeros((len_task,len_task))
        for epoch in range(nb_generations):
            offsprings = Population(
                nb_inds_tasks= [0] * len(self.tasks),
                dim = self.dim_uss, 
                bound = bound,
                list_tasks= self.tasks,
            )
            elite = self.get_elite(population.ls_subPop,20)
            measurement = np.zeros((len_task,len_task))
            for i in range(len_task):
                for j in range(len_task):
                    if i != j :
                        for inv1 in range(20):
                            for inv2 in range(20):
                                measurement[i,j] +=self.distance(elite[i][inv1].genes,elite[j][inv2].genes)
                        measurement[i,j] = 1 / measurement[i,j]
                            # measurement[i,j] *= (1+ 1 / (inv1+1))
            # measurement = self.get_pop_intersection_v2(elite)
            IM = self.get_pop_intersection_v2(elite)
            for i in range(len_task):
                sum_tmp = np.sum(measurement[i])
                for j in range(len_task):
                    if i !=j:
                        rmp[i,j] = measurement[i,j] / sum_tmp *inter[i]
                rmp[i,i]= intra[i]
            # create new offspring population 
            for i in range(len_task):
                    while(len(population.ls_subPop[i])+len(offsprings.ls_subPop[i]) < 2* current_inds_each_task[i]):
                        if np.random.rand() < 0.3:
                            pa = population.__getIndsTask__(idx_task=i,type='tournament',tournament_size=2 )
                            # pa1= self.findParentSameSkill(population[pa.skill_factor],pa)
                            # oa,_=self.crossover(pa,pa1)
                            oa = self.mutation(pa,return_newInd=True)
                            oa.skill_factor = pa.skill_factor
                            population.__addIndividual__(oa)
                        
                        else:
                            k =self.RoutletWheel(IM[i],np.random.rand())
                            pa = population.__getIndsTask__(idx_task=i,type='tournament',tournament_size=2 )
                            pb = population.__getIndsTask__(idx_task=k ,type='tournament',tournament_size=2)
                            oa,ob = self.crossover(pa,pb)
                            oa.skill_factor =pa.skill_factor
                            ob.skill_factor = pa.skill_factor
                            offsprings.__addIndividual__(oa)
                            offsprings.__addIndividual__(ob)
                            if i!=k:
                                oa.transfer =True
                                ob.transfer =True

            offsprings.update_rank()                
            elite_off = self.get_elite(offsprings.ls_subPop,50)
            for i in range(len(self.tasks)):
                x = 0 #inter
                y= 0 #intra
                for j in range(50):
                    if elite_off[i][j].transfer: 
                        x+=1
                    else :y+=1
                y = y/(x+y)
                y = max(0.1,min(0.9,y))
                x = 1-y

                   
                inter[i] =0.5*inter[i]+x*0.5
                intra[i] =0.5*intra[i]+y*0.5
            # merge and update rank
            population = population + offsprings 
            population.update_rank()

            # selection 
            self.selection(population, current_inds_each_task)
           

            # save history 
            self.history_cost.append([ind.fcost for ind in population.get_solves()])
            
            self.render_process((epoch + 1)/nb_generations, ["Cost"], [self.history_cost[-1]], use_sys= True)
            self.IM.append(np.copy(IM))
            self.rmp_hist.append(np.copy(rmp))
        print("End")
        # solve 
        self.last_pop = population 
        return self.last_pop.get_solves()




