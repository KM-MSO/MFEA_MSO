from dataclasses import replace
import numpy as np
from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.task import AbstractTask
from ..EA import *
import time
class model(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: List[AbstractTask], 
        crossover: Crossover.SBX_Crossover, mutation: Mutation.Polynomial_Mutation, selection: Selection.ElitismSelection, 
        *args, **kwargs):
        return super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)
    def manhattan(self,a,b):
        return np.sum(np.abs(a-b))
    def distance(self,population):
        r = np.zeros(len(population))
        sum = np.zeros(population.dim)
        for i in range(len(population)):
            sum += population[i].genes
        for i in range(len(population)):
            r[i] = self.manhattan(population[i].genes*len(population),sum)
        return r/np.sum(r)
    
     
    def transfer_spark(self, population: Population, target_task:int, other_task:int, firework: Individual, 
        sigma: float, nb_inds_each_task: int, rank: int, alpha: float ):
        
        # Compute transfer vector TV that contain the information from other task to target task
        TV_gene = np.zeros_like(firework.genes)
        for i in range(nb_inds_each_task):
            if population[other_task].factorial_rank[i] < sigma * nb_inds_each_task:
                TV_gene += population[other_task][i].genes
        for i in range(nb_inds_each_task):
            if population[target_task].factorial_rank[i] < sigma * nb_inds_each_task:
                TV_gene -= population[target_task][i].genes
        denom_temp = 0
        for r in range(1, nb_inds_each_task + 1):
            denom_temp += (r**-alpha)
        TV_gene = TV_gene * 2 / (sigma * (nb_inds_each_task + nb_inds_each_task)) * (rank ** -alpha) / denom_temp

        # Generate transfer spark
        TS_gene = firework.genes + TV_gene
        TS = self.IndClass(TS_gene, parent = firework)
        TS.skill_factor = other_task
        return TS
    def fit(self, nb_generations, rmp = 0.3, nb_inds_each_task = 100, evaluate_initial_skillFactor = True, *args, **kwargs) -> List[Individual]:
        super().fit(*args, **kwargs)
        # params 
        Cr = 0.9
        Ca= 1.2
        sigma = 0.2
        # initialize population
        population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )

        # save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])
        
        self.render_process(0, ['Cost'], [self.history_cost[-1]], use_sys= True)
        for epoch in range(nb_generations):
            # initial offspring_population of generation
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks = [0] * len(self.tasks), 
                dim = self.dim_uss,
                list_tasks= self.tasks,
            )
            #update Guiding Spark
            # create offspring pop
            for k in range(len(self.tasks)):
                #update Guiding Spark
                delta = np.zeros(self.dim_uss)
                for j in range(0,int(len(population.ls_subPop[k])* sigma)):
                    delta -=population.ls_subPop[k][j].genes
                for j in range(nb_inds_each_task-int(nb_inds_each_task*sigma)+1,nb_inds_each_task):
                    delta+= population.ls_subPop[k][j].genes
                #generate spark
                for j in range(len(population.ls_subPop[k])):
                    if population[k][j].parent is not None:
                        if population[k][j].fcost < population[k][j].parent.fcost:
                            population[k][j].A = population[k][j].A*Cr
                        else: 
                            population[k][j].A = population[k][j].A*Ca
                    gene = population[k][j].genes + population[k][j].A*np.random.choice([-1,1])
                    gene = np.clip(gene,0,1)
                    oa = self.IndClass(genes = gene,parent = population[k][j])
                    oa.skill_factor = k
                    oa.fcost = self.tasks[k](gene)
                    gene = population[k][j].genes+delta
                    gene = np.clip(gene,0,1)
                    ob = self.IndClass(genes = gene,parent = population[k][j])
                    ob.skill_factor = k
                    ob.fcost = self.tasks[k](gene)
                    offsprings.__addIndividual__(oa)
                    offsprings.__addIndividual__(ob)
                #update transfer spark
                    for other_t in range(len(self.tasks)):
                        if other_t != k:
                            TP = self.transfer_spark(
                                population=population,
                                target_task=k,
                                other_task=other_t,
                                firework=population[k][j],
                                sigma=sigma,
                                nb_inds_each_task=nb_inds_each_task,
                                rank=population[k].factorial_rank[j],
                                alpha=0
                                )
                            TP.eval(self.tasks[other_t])
                            offsprings.__addIndividual__(TP)
                            
            # merge and update rank
            population = population + offsprings
            population.update_rank()
            for k in range(len(self.tasks)):
                r = self.distance(population.ls_subPop[k])
                list_selected = np.random.choice(len(population.ls_subPop[k]),(nb_inds_each_task),p = r,replace = False)
                if int(np.argmin(population.ls_subPop[k].factorial_rank)) not in list_selected:
                    rand = np.random.randint(nb_inds_each_task)
                    list_selected[rand] = int(np.argmin(population.ls_subPop[k].factorial_rank))
                population.ls_subPop[k].select(list_selected)
            # save history
            self.history_cost.append([ind.fcost for ind in population.get_solves()])

            #print
            self.render_process((epoch+1)/nb_generations, ['Cost'], [self.history_cost[-1]], use_sys= True)
        print('\nEND!')

        #solve
        self.last_pop = population
        return self.last_pop.get_solves() 
