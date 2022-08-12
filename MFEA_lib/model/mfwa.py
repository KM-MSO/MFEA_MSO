from copy import deepcopy
import numpy as np
from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.task import AbstractTask
from ..EA import *

class model(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: List[AbstractTask], 
        crossover: Crossover.SBX_Crossover, mutation: Mutation.Polynomial_Mutation, selection: Selection.ElitismSelection, 
        *args, **kwargs):
        return super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)
    
    def calculate_num_spark(self, S_param, rank, alpha, N):
        denom = 0
        for r in range(1, N+1):
            denom += r**(-alpha)
        num_spark = int(S_param * ((rank+1)**(-alpha)) / denom)
        # print(f"S_param: {S_param} - rank: {rank+1} - denom: {denom} - num_spark: {num_spark}")
        return num_spark

    def explosion_spark(self, firework: Individual, amplitude: float):
        spark = Individual(firework.genes, dim=len(firework.genes))
        for d in range(len(firework.genes)):
            spark.genes[d] = spark.genes[d] + amplitude * np.random.uniform(low=-1, high=1)
        spark.skill_factor = firework.skill_factor
        return spark
    
    def guiding_spark(self, firework:Individual, sparks: SubPopulation, threshold:float):
        delta = np.zeros_like(firework.genes)
        for i in range(len(sparks)):
            if sparks.factorial_rank[i] < threshold:
                delta += sparks[i].genes
            else:
                delta -= sparks[i].genes
        delta /= threshold
        gene = firework.genes + delta
        gs = Individual(gene, len(gene))
        gs.skill_factor = firework.skill_factor
        return gs
    
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
        TS = Individual(TS_gene, len(TS_gene))
        TS.skill_factor = other_task
        return TS

    def fit(self, nb_generations=100, rmp = 0.3, nb_inds_each_task = 100, evaluate_initial_skillFactor = True,
        diameter=1, Cr=0.9, Ca=1.2, sigma=0.2, alpha=0, S_param=50,
        *args, **kwargs) -> List[Individual]:
        super().fit(*args, **kwargs)

        # Initialize population
        population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )

        # Save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])
        self.render_process(0, ['Cost'], [self.history_cost[-1]], use_sys= True)

        num_task = len(self.tasks)
        num_spark = np.full(shape=(num_task, nb_inds_each_task), fill_value=0, dtype=int)
        amplitude = np.full(shape=(num_task, nb_inds_each_task), fill_value=diameter, dtype=float)

        archive_pop = deepcopy(population)
        # Loop until reaching stop criteria
        for epoch in range(nb_generations):
            # Initial offspring population that contain sparks, guiding sparks, transfer sparks
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks = [0] * len(self.tasks), 
                dim = self.dim_uss,
                list_tasks= self.tasks,
            )
            transfer_pop = Population(
                self.IndClass,
                nb_inds_tasks = [0] * len(self.tasks), 
                dim = self.dim_uss,
                list_tasks= self.tasks,
            )
            # Generate offsprings
            for t in range(num_task):
                for i in range(nb_inds_each_task):
                    firework = population[t][i]
                    rank_firework = population[t].factorial_rank[i]
                    
                    # Calculate number of sparks and explosion amplitude
                    num_spark[t, i] = self.calculate_num_spark(S_param, rank_firework, alpha, nb_inds_each_task)
                    if epoch > 0:
                        # Compare firework at previous generation that have same rank with current firework to adjust amplitude
                        id = np.where(archive_pop[t].factorial_rank == rank_firework)[0][0]
                        pre_firework = archive_pop[t][id]
                        if firework.fcost > pre_firework.fcost:
                            amplitude[t,i] *= Cr
                        else:
                            amplitude[t,i] *= Ca

                    # Eplosion spark
                    for s in range(num_spark[t, i]):
                        spark = self.explosion_spark(firework=firework, amplitude=amplitude[t,i])
                        spark.eval(self.tasks[t])
                        offsprings.__addIndividual__(spark)

                    # Guiding spark
                    offsprings[t].update_rank()
                    threshold = int(sigma*num_spark[t,i])
                    GS = self.guiding_spark(firework, offsprings[t], threshold)
                    GS.eval(self.tasks[t])
                    offsprings.__addIndividual__(GS)

                    # Transfer spark
                    for other_t in range(num_task):
                        if other_t != t:
                            TP = self.transfer_spark(
                                population=population,
                                target_task=t,
                                other_task=other_t,
                                firework=firework,
                                sigma=sigma,
                                nb_inds_each_task=nb_inds_each_task,
                                rank=rank_firework,
                                alpha=alpha
                                )
                            TP.eval(self.tasks[other_t])
                            transfer_pop.__addIndividual__(TP)


            # print(f"Gen {epoch}")
            # print(num_spark)
            archive_pop = deepcopy(population)

            # Merge and update rank
            population = population + offsprings
            population.update_rank()

            # Keep the candidates
            self.selection(population, [nb_inds_each_task] * len(self.tasks))


            # Merge with transfer spark and selection
            population = population + transfer_pop
            population.update_rank()
            self.selection(population, [nb_inds_each_task] * len(self.tasks))

            # save history
            self.history_cost.append([ind.fcost for ind in population.get_solves()])

            #print
            self.render_process((epoch+1)/nb_generations, ['Cost'], [self.history_cost[-1]], use_sys= True)
        print('\nEND!')

        #solve
        self.last_pop = population
        return self.last_pop.get_solves() 
