import enum
import matplotlib.pyplot as plt
import numpy as np

from ..EA import *
from ..operators import Crossover, Mutation, Search, Selection
from ..tasks.task import AbstractTask
from . import AbstractModel


class model(AbstractModel.model):
    class battle_smp:
        def __init__(self, idx_host: int, nb_tasks: int, lr, p_const_intra) -> None:
            assert idx_host < nb_tasks
            self.idx_host = idx_host
            self.nb_tasks = nb_tasks

            #value const for intra
            self.p_const_intra = p_const_intra
            self.lower_p = 0.1/(self.nb_tasks + 1)

            # smp without const_val of host
            self.sum_not_host = 1 - 0.1 - p_const_intra
            self.SMP_not_host: np.ndarray = ((np.zeros((nb_tasks + 1, )) + self.sum_not_host)/(nb_tasks + 1))
            self.SMP_not_host[self.idx_host] += self.sum_not_host - np.sum(self.SMP_not_host)

            self.SMP_include_host = self.get_smp()
            self.lr = lr

        def get_smp(self) -> np.ndarray:
            smp_return : np.ndarray = np.copy(self.SMP_not_host)
            smp_return[self.idx_host] += self.p_const_intra
            smp_return += self.lower_p
            return smp_return
        
        def update_SMP(self, Delta_task, count_Delta_tasks):
            '''
            Delta_task > 0 
            '''
            # for idx, delta in enumerate(Delta_task):
            #     self.smp_not_host[idx] += (delta / (self.smp_include_host[idx] / self.lower_p)) * self.lr

            if np.sum(Delta_task) != 0:         
                # newSMP = np.array(Delta_task) / (self.SMP_include_host)
                newSMP = (np.array(Delta_task) / (np.array(count_Delta_tasks) + 1e-50))
                newSMP = newSMP / (np.sum(newSMP) / self.sum_not_host + 1e-50)

                self.SMP_not_host = self.SMP_not_host * (1 - self.lr) + newSMP * self.lr
                
                self.SMP_not_host[self.idx_host] += self.sum_not_host - np.sum(self.SMP_not_host)

                self.SMP_include_host = self.get_smp()
            return self.SMP_include_host
    
    def __init__(self, seed=None, percent_print=2) -> None:
        super().__init__(seed, percent_print)
        self.ls_attr_avg.append('history_smp')

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
    
    def render_smp(self,  shape = None, title = None, figsize = None, dpi = 100, step = 1, re_fig = False, label_shape= None, label_loc= None):
        
        if title is None:
            title = self.__class__.__name__
        if shape is None:
            shape = (int(np.ceil(len(self.tasks) / 3)), 3)
        else:
            assert shape[0] * shape[1] >= len(self.tasks)

        if label_shape is None:
            label_shape = (1, len(self.tasks))
        else:
            assert label_shape[0] * label_shape[1] >= len(self.tasks)

        if label_loc is None:
            label_loc = 'lower center'

        if figsize is None:
            figsize = (shape[1]* 6, shape[0] * 5)

        fig = plt.figure(figsize= figsize, dpi = dpi)
        fig.suptitle(title, size = 15)
        fig.set_facecolor("white")
        fig.subplots(shape[0], shape[1])

        his_smp:np.ndarray = np.copy(self.history_smp)
        y_lim = (-0.1, 1.1)

        for idx_task, task in enumerate(self.tasks):
            fig.axes[idx_task].stackplot(
                np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])),
                [his_smp[
                    np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])), 
                    idx_task, t] for t in range(len(self.tasks) + 1)],
                labels = ['Task' + str(i + 1) for i in range(len(self.tasks))] + ["mutation"]
            )
            # plt.legend()
            fig.axes[idx_task].set_title('Task ' + str(idx_task + 1) +": " + task.name)
            fig.axes[idx_task].set_xlabel('Generations')
            fig.axes[idx_task].set_ylabel("SMP")
            fig.axes[idx_task].set_ylim(bottom = y_lim[0], top = y_lim[1])


        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(lines, labels, loc = label_loc, ncol = label_shape[1])
        plt.show()
        if re_fig:
            return fig

    def fit(self, nb_generations: int, nb_inds_each_task: int, nb_inds_min = None,
        lr = 1, p_const_intra = 0.5, swap_po = True, prob_search = 0.5,
        nb_epochs_stop = 50, 
        evaluate_initial_skillFactor = False,
        *args, **kwargs):
        super().fit(*args, **kwargs)
        
        self.p_const_intra = p_const_intra

        # nb_inds_min
        if nb_inds_min is not None:
            assert nb_inds_each_task >= nb_inds_min
        else: 
            nb_inds_min = nb_inds_each_task

        # initial history of smp -> for render
        self.history_smp = []

        #initialize population
        population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )

        nb_inds_tasks = [nb_inds_each_task] * len(self.tasks)
        
        # SA params:
        MAXEVALS = nb_generations * nb_inds_each_task * len(self.tasks)
        eval_k = np.zeros(len(self.tasks))
        epoch = 0

        '''
        ------
        per params
        ------
        '''
        # prob choose first parent
        p_choose_father = np.ones((len(self.tasks), ))/ len(self.tasks)
        # count_eval_stop: nums evals not decrease factorial cost
        # maxcount_es: max of count_eval_stop
        # if count_eval[i] == maxcount_es: p_choose_father[i] == 0
        count_eval_stop = [0] * len(self.tasks)
        maxcount_es = nb_epochs_stop * nb_inds_each_task

        # Initialize memory M_smp
        M_smp = [self.battle_smp(i, len(self.tasks), lr, p_const_intra) for i in range(len(self.tasks))]

        #save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])
        self.history_smp.append([M_smp[i].get_smp() for i in range(len(self.tasks))])
        epoch = 1

        while np.sum(eval_k) <= MAXEVALS:
            turn_eval = [0] * len(self.tasks)

            # initial offspring_population of generation
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks= [0] * len(self.tasks),
                dim =  self.dim_uss, 
                list_tasks= self.tasks,
            )

            # Delta epoch
            Delta:list[list[float]] = np.zeros((len(self.tasks), len(self.tasks) + 1)).tolist()
            count_Delta: list[list[float]] = np.zeros((len(self.tasks), len(self.tasks) + 1)).tolist()

            while np.sum( turn_eval) < np.sum(nb_inds_tasks):
                if np.sum(eval_k) >= epoch * nb_inds_each_task * len(self.tasks):
                    # save history
                    self.history_cost.append([ind.fcost for ind in population.get_solves()])
                    self.history_smp.append([M_smp[i].get_smp() for i in range(len(self.tasks))])

                    self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True)

                    # update mutation
                    self.mutation.update(population = population)

                    epoch += 1

                # choose subpop of father pa
                skf_pa = np.random.choice(np.arange(len(p_choose_father)), p= p_choose_father)

                # get smp 
                smp = M_smp[skf_pa].get_smp()

                # choose subpop of mother pb
                skf_pb = np.random.choice(np.arange(len(smp)), p= smp)

                if skf_pb != len(self.tasks):
                    pa = population[skf_pa].__getRandomItems__()
                    pb = population[skf_pb].__getRandomItems__()
                    while pb is pa:
                        pb = population[skf_pb].__getRandomItems__()

                    if np.all(pa.genes == pb.genes):
                        pb = population[skf_pb].__getWorstIndividual__
                        
                    if pa < pb:
                        pa, pb = pb, pa
                
                    oa, ob = self.crossover(pa, pb, skf_pa, skf_pa)
                else:
                    pa, pb = population.__getIndsTask__(skf_pa, type= 'random', size= 2)
                    if pa < pb:
                        pa, pb = pb, pa

                    oa = self.mutation(pa, return_newInd= True)
                    oa.skill_factor = skf_pa

                    ob = self.mutation(pb, return_newInd= True)
                    ob.skill_factor = skf_pa

                count_Delta[skf_pa][skf_pb] += 2

                # add oa, ob to offsprings population and eval fcost
                offsprings.__addIndividual__(oa)
                offsprings.__addIndividual__(ob)
                
                eval_k[skf_pa] += 2
                turn_eval[skf_pa] += 2

                # Calculate the maximum improvement percetage
                Delta1 = (pa.fcost - oa.fcost)/(pa.fcost + 1e-50)**2
                Delta2 = (pa.fcost - ob.fcost)/(pa.fcost + 1e-50)**2

                # update smp
                if Delta1 > 0 or Delta2 > 0:
                    Delta[skf_pa][skf_pb] += max([Delta1, 0]) ** 2
                    Delta[skf_pa][skf_pb] += max([Delta2, 0]) ** 2

                    # swap
                    if swap_po:
                        if Delta1 > Delta2:
                            # swap oa (-2) with pa 
                            offsprings[skf_pa].ls_inds[-2], population[skf_pa].ls_inds[population[skf_pa].ls_inds.index(pa)] = pa, oa
                            if Delta2 > 0 and skf_pa == skf_pb:
                                #swap ob (-1) with pb 
                                offsprings[skf_pa].ls_inds[-1], population[skf_pa].ls_inds[population[skf_pa].ls_inds.index(pb)] = pb, ob
                        else:
                            #swap ob (-1) with pa 
                            offsprings[skf_pa].ls_inds[-1], population[skf_pa].ls_inds[population[skf_pa].ls_inds.index(pa)] = pa, ob
                            if Delta1 > 0 and skf_pa == skf_pb:
                                offsprings[skf_pa].ls_inds[-2], population[skf_pa].ls_inds[population[skf_pa].ls_inds.index(pb)] = pb, oa
                    # reset count_eval_stop
                    count_eval_stop[skf_pa] = 0
                else:
                    # count eval not decrease cost
                    count_eval_stop[skf_pa] += 1

                if count_eval_stop[skf_pa] > maxcount_es:
                    Delta[skf_pa][len(self.tasks)] += 1e-5
                elif count_eval_stop[skf_pa] == 0:
                    pass
                    

            # merge
            population = population + offsprings
            population.update_rank()

            # selection
            nb_inds_tasks = [int(
                # (nb_inds_min - nb_inds_each_task) / nb_generations * (epoch - 1) + nb_inds_each_task
                int(min((nb_inds_min - nb_inds_each_task)/(nb_generations - 1)* (epoch - 1) + nb_inds_each_task, nb_inds_each_task))
            )] * len(self.tasks)
            self.selection(population, nb_inds_tasks)

            # update operators
            self.crossover.update(population = population)
            self.mutation.update(population = population)

            # update smp
            for skf in range(len(self.tasks)):
                M_smp[skf].update_SMP(Delta[skf], count_Delta[skf])
                
            '''
            search
            '''
            for subPop in population:
                for idx in range(len(subPop)):
                    if np.random.rand() < prob_search:
                        '''
                        DE
                        '''
                        new_ind = self.search(ind = subPop[idx], population = population)
                        if new_ind.fcost < subPop[idx].fcost:
                            # subPop.__addIndividual__(new_ind)
                            subPop.ls_inds[idx] = new_ind
                            subPop.update_rank()
                        eval_k[subPop.skill_factor] += 1
                        turn_eval[subPop.skill_factor] += 1
            self.search.update()

            '''
            multistart and sample 
            '''
            for skf in range(len(self.tasks)): 
                # if count_eval_stop[skf] > maxcount_es:
                if epoch % kwargs['step_over'] == 0: 
                    '''
                    sample 1/2 subpopulation
                    '''
                    # take the cov
                    matrix = [] 
                    for idx, ind in enumerate(population[skf]): 
                        if population[skf].factorial_rank[idx] <= len(population[skf].factorial_rank) / 2 :
                            matrix.append(ind.genes) 
                    matrix = np.array(matrix).T

                    # matrix = np.array([ind.genes for ind in population[skf]]).T 
                    cov = np.cov(matrix)
                    mean = np.mean(matrix, axis= 1) 

                    # sample 
                    for idx, ind in enumerate(population[skf]): 
                        if population[skf].factorial_rank[idx] > len(population[skf].factorial_rank) / 2 : 
                            bf_shape= ind.genes.shape 
                            ind.genes = np.random.multivariate_normal(mean = mean, cov = cov) 
                            assert ind.genes.shape == bf_shape
                            ind.fcost= ind.eval(self.tasks[skf]) 
                            eval_k[skf] += 1 
                    
                    population[skf].update_rank() 
                

            
        #solve
        self.last_pop = population
        self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True)
        print()
        print(p_choose_father)
        print(eval_k)
        print('END!')
        return self.last_pop.get_solves()
    