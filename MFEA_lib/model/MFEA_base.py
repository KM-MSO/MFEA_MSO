import numpy as np
from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.task import AbstractTask
from ..EA import *
from ..tasks.surrogate import GraphDataset

class model(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: list[AbstractTask], 
        crossover: Crossover.SBX_Crossover, mutation: Mutation.Polynomial_Mutation, selection: Selection.ElitismSelection, 
        *args, **kwargs):
        return super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)
    
    def fit(self, nb_generations, rmp = 0.3, nb_inds_each_task = 100, evaluate_initial_skillFactor = True, *args, **kwargs) -> list[Individual]:
        super().fit(*args, **kwargs)

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

            # create offspring pop
            while len(offsprings) < len(population):
                # choose parent 
                pa, pb = population.__getRandomInds__(2)

                if pa.skill_factor == pb.skill_factor or np.random.rand() < rmp:
                    # intra / inter crossover
                    skf_oa, skf_ob = np.random.choice([pa.skill_factor, pb.skill_factor], size= 2, replace= True)
                    oa, ob = self.crossover(pa, pb, skf_oa, skf_ob)
                else:
                    # mutate
                    oa = self.mutation(pa, return_newInd= True)
                    oa.skill_factor = pa.skill_factor

                    ob = self.mutation(pb, return_newInd= True)    
                    ob.skill_factor = pb.skill_factor
                
                offsprings.__addIndividual__(oa)
                offsprings.__addIndividual__(ob)
            
            # merge and update rank
            population = population + offsprings
            population.update_rank()

            # selection
            self.selection(population, [nb_inds_each_task] * len(self.tasks))

            # save history
            self.history_cost.append([ind.fcost for ind in population.get_solves()])

            #print
            self.render_process((epoch+1)/nb_generations, ['Cost'], [self.history_cost[-1]], use_sys= True)
        print('\nEND!')

        #solve
        self.last_pop = population
        return self.last_pop.get_solves() 

class betterModel(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: list[AbstractTask], 
        crossover: Crossover.SBX_Crossover, mutation: Mutation.Polynomial_Mutation, selection: Selection.ElitismSelection, 
        *args, **kwargs):
        if 'surrogate_pipeline' in kwargs.keys():
            self.surrogate_pipeline = kwargs.get('surrogate_pipeline')
            self.dataset = GraphDataset(tasks = tasks)
        return super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)
    
    def fit(self, nb_generations, rmp = 0.3, nb_inds_each_task = 100, evaluate_initial_skillFactor = True, *args, **kwargs) -> list[Individual]:
        super().fit(*args, **kwargs)

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
            genes, costs, skf = self.epoch_step(rmp, epoch, nb_inds_each_task, nb_generations, population)
            self.dataset.append(genes, costs,skf)
        print('\nEND!')

        #solve
        self.last_pop = population
        return self.last_pop.get_solves() 
    
    
    def epoch_step(self, rmp, cur_epoch, nb_inds_each_task, nb_generations, population):
        # initial offspring_population of generation
        offsprings = Population(
            self.IndClass,
            nb_inds_tasks = [0] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
        )

        # create offspring pop
        while len(offsprings) < len(population):
            # choose parent 
            pa, pb = population.__getRandomInds__(2)

            if pa.skill_factor == pb.skill_factor or np.random.rand() < rmp:
                # intra / inter crossover
                skf_oa, skf_ob = np.random.choice([pa.skill_factor, pb.skill_factor], size= 2, replace= True)
                oa, ob = self.crossover(pa, pb, skf_oa, skf_ob)
            else:
                # mutate
                oa = self.mutation(pa, return_newInd= True)
                oa.skill_factor = pa.skill_factor

                ob = self.mutation(pb, return_newInd= True)    
                ob.skill_factor = pb.skill_factor
            
            offsprings.__addIndividual__(oa)
            offsprings.__addIndividual__(ob)
        
        # merge and update rank
        population = population + offsprings
        
        population.update_rank()

        # for subpop in population.ls_subPop:
        #     print(np.min([ind.fcost for ind in subpop.ls_inds]))
            
        # selection
        self.selection(population, [nb_inds_each_task] * len(self.tasks))
        # save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])
        
        sol = population.get_all_inds()

        #print
        self.render_process((cur_epoch+1)/nb_generations, ['Cost'], [self.history_cost[-1]], use_sys= True)
        return np.stack([ind.genes for ind in sol]), np.hstack([ind.fcost for ind in sol]), np.hstack([ind.skill_factor for ind in sol])