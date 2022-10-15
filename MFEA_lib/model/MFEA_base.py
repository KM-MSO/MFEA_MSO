from re import A
import numpy as np
from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.task import AbstractTask
from ..EA import *
from ..tasks.surrogate import GraphDataset
import random
from scipy.stats import kendalltau
from sklearn.metrics import f1_score, confusion_matrix
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
    
    def fit(self, nb_generations, rmp = 0.3, nb_inds_each_task = 100, evaluate_initial_skillFactor = True, train_period = 5, start_eval = 6, *args, **kwargs) -> list[Individual]:
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
            genes, costs, skf, bests, population= self.epoch_step(rmp, epoch, nb_inds_each_task, nb_generations, population)
            self.dataset.append(genes, costs,skf, bests)
            if epoch + 1 > start_eval:
                reg_preds = []
                cls_preds = []
                cls_gts = []
                gts = []
                for d in self.dataset.latest_data:
                    reg_predict, cls_predict = self.surrogate_pipeline.predict(d)
                    reg_preds.append(reg_predict.detach().cpu().numpy()[0])
                    cls_preds.append(1 if cls_predict.detach().cpu().numpy()[0] > 0.5 else 0)
                    cls_gts.append(d.thresh_hold.cpu().numpy()[0])
                    gts.append(d.y.cpu().numpy()[0])
                
                print(kendalltau(reg_preds, gts), f'F1: {f1_score(cls_gts, cls_preds)}')
                print(confusion_matrix(cls_gts, cls_preds))
            if (epoch + 1) % train_period == 0:
                self.surrogate_pipeline.train(self.dataset)
            
                
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
        
        # population.update_rank()
        population.update_rank()
        sol = random.sample(offsprings.get_all_inds(), 100)
        # selection
        self.selection(population, [nb_inds_each_task] * len(self.tasks))
        last_best = np.stack([self.history_cost[-1][ind.skill_factor] for ind in sol])
        # save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])

        for i in range(len(self.history_cost[-1])):
            assert self.history_cost[-1][i] == np.min([f.fcost for f in population.ls_subPop[i].ls_inds]), (self.history_cost[-1][i] , np.min([f.fcost for f in population.ls_subPop[i].ls_inds]))

        #print
        self.render_process((cur_epoch+1)/nb_generations, ['Cost'], [self.history_cost[-1]], use_sys= True)
        return np.stack([ind.genes for ind in sol]), np.hstack([ind.fcost for ind in sol]), \
                np.hstack([ind.skill_factor for ind in sol]), last_best, population