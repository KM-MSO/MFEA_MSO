from asyncio import tasks
from operator import itemgetter
# from xxlimited import new
# from re import U, sub
import numpy as np
import operator
import scipy.stats

from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.function import AbstractFunc
from ..EA import *
import matplotlib.pyplot as plt
import copy

class model(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: list[AbstractTask], 
        crossover: Crossover.SBX_Crossover, mutation: Mutation.GaussMutation, selection: Selection.ElitismSelection, 
        *args, **kwargs):
        return super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)
    
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
    def RoutletWheel(self,rmp,rand, max):
        tmp = [0]*len(self.tasks)
        tmp[0]= rmp[0]
        for i in range(1,len(tmp)):
            tmp[i]=tmp[i-1]+rmp[i]
        index =0 
        while tmp[index] < rand:
            index+=1
            if index == max - 1:
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

    def Add_Archie (archie, index, genes) :
        archie[index] = np.copy(genes)

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
        H = 6
        population.update_rank()
        #history
        # self.rmp_hist = []
        len_task  = len(self.tasks)
        #SA param
        nb_inds_tasks = [nb_inds_each_task]*len(self.tasks)
        MAXEVALS = nb_generations * nb_inds_each_task * len(self.tasks)
        epoch =0 
        eval_k = np.zeros(len(self.tasks))
        archie = []
        for i in range(len_task) :
            new_archie = []
            archie.append(new_archie)
        for i in range(len_task) :
            for j in range(nb_inds_each_task) :
                archie[i].append(np.copy(population[i].ls_inds[j].genes))
        for i in range(len_task) :
            for inv in population[i].ls_inds :
                inv.init_gen = 0
        u_CR = 0.5 * np.ones([len_task, 2])
        u_F = 0.5 * np.ones([len_task,2])
        u_mt = 0.5 * np.ones(len_task)
        u_rmp = 0.5*np.ones([len_task, len_task])
        m_CR = 0.5 * np.ones([len_task,2,H])
        m_F = 0.5 * np.ones([len_task,2,H])
        m_rmp = 0.5 * np.ones([len_task,len_task,H])
        m_mt = 0.5*np.ones([len_task,H]) 
        len_archie = 600
        smp = np.zeros([len_task, len_task])
        for i in range(len_task) :
            for j in range(len_task) :
                if i==j :
                    smp[i,j] = 0.5
                else :
                    smp[i,j] = 0.5 / (len_task - 1)
        self.u_CR = []
        self.u_F = []
        self.smp = []
        # best_fitness = np.zeros(len_task)
        # deadlock = np.zeros(len_task)

        while np.sum(eval_k) <= MAXEVALS: 
            # if np.sum(eval_k) >= epoch * nb_inds_each_task * len(self.tasks):
                    # save history 
            self.history_cost.append([ind.fcost for ind in population.get_solves()])
                
                # self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(u) for u in population.ls_subPop], self.history_cost[-1]], use_sys= True)
            self.render_process(np.sum(eval_k)/MAXEVALS, ['Pop_size', 'Cost'], [[len(u) for u in population.ls_subPop], self.history_cost[-1]], use_sys= True)

                # self.IM.append(np.copy(IM))
                # self.rmp_hist.append(np.copy(rmp))
            epoch = (epoch + 1)%1000 
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks= [0] * len(self.tasks),
                dim =  self.dim_uss, 
                list_tasks= self.tasks,
            )

            best = self.get_elite(population.ls_subPop, 11)
            u_CR = np.mean(m_CR, axis = 2)
            u_F = np.mean(m_F, axis = 2)
            u_mt = np.mean(m_mt, axis = 1)
            u_rmp = np.mean(m_rmp, axis = 2)
            # print(len(population[i].ls_inds), eval_k[0], epoch)
            for i in range(len_task):
                increment = np.zeros(len_archie)
                num_hybrid = np.zeros(len_archie)
                s_F = [[], []]
                s_CR = [[], []]
                s_mt = []
                s_rmp = []
                delta_f_hybrid = [] 
                for j in range(len_task) :
                    tmp1 = []
                    delta_f_hybrid.append(tmp1)

                    tmp2 = []
                    s_rmp.append(tmp2)
                delta_f_mt = []
                delta_f_DE = [[], []]              
                # for j in range(nb_inds_tasks[i]) :
                #     eval_k[i] += 1
                #     inv = population[i].ls_inds[j]
                j = 0
                for inv in population[i].ls_inds :
                    eval_k[i] += 1
                    # if epoch - inv.init_gen >= 20 and population[i].factorial_rank[j] > 1 :
                    #     child = self.mutation(inv, return_newInd= True)
                    #     child.skill_factor = i
                    #     child.init_gen = epoch + 1
                    #     offsprings.__addIndividual__(child)
                    #     archie[i][index[i]] = np.copy(child.genes)
                    #     index[i] = (index[i] + 1)%500
                    #     continue

                    # k = random.randint(0,len_task - 1)
                    k = self.RoutletWheel(smp[i],random.random(),len_task)
                    num_hybrid[k] += 1
                    # temp_rmp = np.random.normal(loc = u_rmp[i,k], scale = 0.1)
                    if k!=i :
                        eval_k[i] += 1
                        p1 = population[k].__getRandomItems__()
                        child1, child2 = self.crossover(inv, p1, inv.skill_factor, inv.skill_factor)
                        child1.fcost = self.tasks[i](child1.genes)
                        child2.fcost = self.tasks[i](child2.genes)
                        if child1.fcost < child2.fcost :
                            child = child1
                        else :
                            child = child2
                        child.init_gen = epoch + 1
                        if child.fcost < inv.fcost :
                            offsprings.__addIndividual__(child)
                            delta_f_hybrid[k].append((inv.fcost - child.fcost) / inv.fcost)
                            increment[k] += inv.fcost - child.fcost 
                            if len(archie[i]) < len_archie :
                                archie[i].append(np.copy(child.genes))
                            else :
                                rand = random.randint(0,len_archie-1)
                                archie[i][rand] = np.copy(child.genes)
                        else :
                            offsprings.__addIndividual__(inv)
                    else :
                        temp_mt = np.random.normal(loc = u_mt[i], scale = 0.1)
                        while temp_mt < 0 or temp_mt > 1 :
                            temp_mt = np.random.normal(loc = u_mt[i], scale = 0.1)
                        index_DE = 0
                        # if random.random() < temp_mt :
                        if random.random() < 10 :
                            index_DE = 0
                            f = np.random.normal(loc = u_F[i,0], scale = 0.1)
                            f = min(1, max(0,f))
                            cr = scipy.stats.cauchy.rvs(loc = u_CR[i,0], scale = 0.1)
                            cr = min(1, max(cr, 0))

                            p0 = random.randint(0, 9)
                            p1 = random.randint(0, len(archie[i]) - 1)
                            p2 = random.randint(0, len(archie[i]) - 1)
    
                            r1 = archie[i][p1]
                            r2 = archie[i][p2]
                            V_k = inv.genes + f*(best[i][p0].genes - inv.genes) + f*(r1 - r2)
                            V_k = np.clip(V_k, 0, 1)
                        else :
                            index_DE = 1
                            f = np.random.normal(loc = u_F[i,1], scale = 0.1)
                            f = min(1, max(0,f))
                            cr = scipy.stats.cauchy.rvs(loc = u_CR[i,1], scale = 0.1)
                            cr = min(1, max(cr, 0))

                            p1 = random.randint(0, len(archie[i]) - 1)
                            p2 = random.randint(0, len(archie[i]) - 1)
                            p3 = random.randint(0, len(archie[i]) - 1)
                            p4 = random.randint(0, len(archie[i]) - 1)
                            r1 = archie[i][p1]
                            r2 = archie[i][p2]
                            r3 = archie[i][p3]
                            r4 = archie[i][p4]
                            # V_k = inv.genes + f*(r3 - inv.genes) + f*(r1 - r2)
                            V_k = inv.genes + f*(r1 - r2)
                            # V_k = np.clip(V_k, 0, 1)

                        j_rand = random.randint(0, self.dim_uss-1)
                        gen_child = np.zeros(self.dim_uss)
                        for l in range(self.dim_uss) :
                            if random.random() < cr or l == j_rand :
                                gen_child[l] = V_k[l]
                                if gen_child[l] < 0 :
                                    gen_child[l] = (inv.genes[l]) / 2
                                if gen_child[l] > 1 :
                                    gen_child[l] = (inv.genes[l] + 1) / 2
                            else :
                                gen_child[l] = inv.genes[l]       
                        child = self.IndClass(gen_child)
                        child.skill_factor=i
                        child.fcost = self.tasks[i](child.genes)
                        child.init_gen = epoch + 1
                        if child.fcost < inv.fcost :
                            increment[k] += inv.fcost - child.fcost 
                            s_F[index_DE].append(f)
                            s_CR[index_DE].append(cr)
                            s_mt.append(temp_mt)
                            # s_rmp.append(temp_rmp)
                            delta_f_DE[index_DE].append((inv.fcost - child.fcost)/inv.fcost)
                            # delta_f_hybrid.append((inv.fcost - child.fcost)/inv.fcost)
                            delta_f_mt.append((inv.fcost - child.fcost)/inv.fcost)
                        # if child.fcost < inv.fcost or (random.random() < (epoch - inv.init_gen) * 0.015 and population[i].factorial_rank[j] > 5 ) :
                            child.init_gen = epoch + 1
                            child.skill_factor = inv.skill_factor
                            offsprings.__addIndividual__(child)
                            if len(archie[i]) < len_archie :
                                archie[i].append(np.copy(child.genes))
                            else :
                                rand = random.randint(0,len_archie-1)
                                archie[i][rand] = np.copy(child.genes)
                        else :
                            offsprings.__addIndividual__(inv)
                    j+=1
                for j in range(len_task) :
                    if num_hybrid[j] > 0 :
                        increment[j] = increment[j] / num_hybrid[j]
                tmp = np.sum(increment)
                if tmp > 0 :
                    for j in range(len_task) :
                        smp[i,j] = 0.8*smp[i,j] + 0.2*increment[j] / tmp

                for j in range(2) :
                    num_inv_success = len(s_CR[j])
                    if num_inv_success > 0 :
                        sf = np.array(s_F[j])
                        scr = np.array(s_CR[j])
                        deltafDE = np.array(delta_f_DE[j])

                        mean_F = np.sum(sf**2) / np.sum(sf)
                        mean_F = max(0.1, min(mean_F, 1))
                        mean_CR = np.sum(scr * deltafDE) / np.sum(deltafDE)
                        mean_CR = max(0.1, min(mean_CR, 1))
                        # print(mean_F, mean_CR)
                        m_F[i,j,(epoch%H)] = mean_F
                        m_CR[i,j,(epoch%H)] = mean_CR
                # for j in range(len_task) :
                #     if i!=j :
                #         num_inv_success = len(delta_f_hybrid[j]) 
                #         if num_inv_success > 0 :
                #             srmp = np.array(s_rmp[j])
                #             deltafhybrid = np.array(delta_f_hybrid[j])
                #             m_rmp[i,j,(epoch%H)] = np.sum(srmp*srmp*deltafhybrid) / np.sum(srmp*deltafhybrid)
                #             m_rmp[i,j,(epoch%H)] = max (0.01, min(m_rmp[i,j,(epoch%H)],1))
                #         else :
                #             m_rmp[i,j,(epoch%H)] -= 0.025
                #             m_rmp[i,j,(epoch%H)] = max (0.01, min(m_rmp[i,j,(epoch%H)],1))

                num_inv_success = len(delta_f_mt)
                if num_inv_success > 0 :
                    smt = np.array(s_mt)
                    deltafmt = np.array(delta_f_mt)
                    m_mt[i,(epoch%H)] = np.sum(smt*smt*deltafmt) / np.sum(smt*deltafmt)
                    m_mt[i,(epoch%H)] = max (0.1, min(m_mt[i,(epoch%H)],0.9))
                # else :
                #     m_mt[i,(epoch%H)] = 0.5
                
            # print(mutation_rate[0])
            for i in range(len_task) :
                population[i].ls_inds.clear()
                population[i].ls_inds += offsprings[i].ls_inds
            population.update_rank()

            # for i in range(len_task) :
            #     print(len(population[i].ls_inds))
            # print(u_mt)

            # selection 
            if LSA is True:
                # nb_inds_tasks = [int(
                #     # (nb_inds_min - nb_inds_each_task) / nb_generations * (epoch - 1) + nb_inds_each_task
                #     int(min((nb_inds_min - nb_inds_each_task)/(nb_generations - 1)* epoch  + nb_inds_each_task, nb_inds_each_task))
                # )] * len(self.tasks)
                for j in range(len_task) :
                    if epoch%30 == 29 and nb_inds_tasks[j] > 60 :
                        nb_inds_tasks[j] -= 1
            self.selection(population, nb_inds_tasks)
            # if epoch > 0 :
            #     self.mutation.update(population)
            # save history
            self.history_cost.append([ind.fcost for ind in population.get_solves()])
                
            # self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(u) for u in population.ls_subPop], self.history_cost[-1]], use_sys= True)
            self.render_process(np.sum(eval_k)/MAXEVALS, ['Pop_size', 'Cost'], [[len(u) for u in population.ls_subPop], self.history_cost[-1]], use_sys= True)
            # self.IM.append(np.copy(IM))
            # self.rmp_hist.append(np.copy(rmp))
            if np.sum(eval_k) >= MAXEVALS :
                self.render_process(1, ['Pop_size', 'Cost'], [[len(u) for u in population.ls_subPop], self.history_cost[-1]], use_sys= True)
                break
            # print(epoch)
        print("End")
        # solve 
        self.last_pop = population 
        return self.last_pop.get_solves()




