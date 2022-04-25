import numpy as np
import sys
import scipy.stats

DIMENSIONS = 50 


# np.random.seed(1)



def skill_factor_best_task(pop, tasks,):
    
    population = np.copy(pop)
    maxtrix_cost = np.array(
        [np.apply_along_axis(t.func, 1, population) for t in tasks]
    ).T
    matrix_rank_pop = np.argsort(np.argsort(maxtrix_cost, axis=0), axis=0)

    N = len(population) / len(tasks)
    count_inds = np.array([0] * len(tasks))
    skill_factor_arr = np.zeros(
        int(
            (N * len(tasks)),
        ),
        dtype=np.int,
    )
    condition = False

    while not condition:
        idx_task = np.random.choice(np.where(count_inds < N)[0])

        idx_ind = np.argsort(matrix_rank_pop[:, idx_task])[0]

        skill_factor_arr[idx_ind] = idx_task

        matrix_rank_pop[idx_ind] = len(pop) + 1
        count_inds[idx_task] += 1

        condition = np.all(count_inds == N)

    return skill_factor_arr


def cal_factor_cost(population, tasks, skill_factor):
    factorial_cost = np.zeros_like(skill_factor, dtype=float)
    for i in range(len(population)):
        factorial_cost[i] = tasks[skill_factor[i]].func(population[i])

    return factorial_cost

def create_population(number_population, dimension, lower, upper):
    '''
    Arguments: 
        number_population: Number individuals in population 
        dimension : the number of genes in each individual
        lower: lower bound of gene 
        upper: upper bound of gene 
    Returns:
        population: the list of individual
    '''
    population = np.random.uniform(
        low=lower, high=upper, size=(number_population, dimension)
    )
    return population

def add_coefficient_gauss(population): 
    # Thêm vào cuối mỗi phần tử là một số random từ 0 -> 0.1 
    gauss_add_element= np.abs(np.random.normal(loc= 0, scale=0.1, size= (len(population),1)))
    population = np.append(population, gauss_add_element, axis= 1) 
    return population 

def compute_scalar_fitness(factorial_cost, skill_factor):
    """
    Compute scalar fitness for individual in its task 

    Arguments: 
        factorial_cost: np.array(size population,) factorial cost of each individual 
        skill_factor: np.array(size population, ) skill_factor of each individual 

    Returns:
        Scalar fitness: np.array(size population, ) 1/rank of individual of its task. 
    """
    number_task = np.max(skill_factor) + 1

    number_population = len(factorial_cost)
    temp = [[] for i in range(number_task)]
    index = [[] for i in range(number_task)]
    scalar_fitness = np.zeros_like(skill_factor, dtype=float)
    for ind in range(number_population):
        task = skill_factor[ind]
        temp[task].append(factorial_cost[ind])
        index[task].append(ind)

    for task in range(number_task):
        index_sorted = np.argsort(np.array(temp[task]))
        for order in range(len(index_sorted)):
            scalar_fitness[index[task][index_sorted[order]]] = 1.0 / float(1.0 + order)

    return scalar_fitness

def lsa_SaMTPSO_DE_SBX_new_gauss(tasks, lsa=True, seed = 1, rate= 0.5, max_popu = 100, min_popu= 20, ti_le_giu_lai = 0.9):

    np.random.seed(seed)
    MAXEVALS = 1000 * 100 * len(tasks)
    DIMENSIONS = 50 
    LOWER_BOUND = 0
    UPPER_BOUND = 1 
    initial_size_population = np.zeros((len(tasks)), dtype=int) + max_popu
    current_size_population = np.copy(initial_size_population)
    min_size_population = np.zeros((len(tasks)), dtype=int) + min_popu

    evaluations = np.zeros((len(tasks)), dtype=int)
    maxEvals = np.zeros_like(evaluations, dtype=int) + int(MAXEVALS / len(tasks))

    skill_factor = np.zeros((np.sum(initial_size_population)), dtype=int)
    factorial_cost = np.zeros((np.sum(initial_size_population)), dtype=float)
    population = create_population(
        np.sum(initial_size_population), DIMENSIONS, LOWER_BOUND, UPPER_BOUND
    )

    
    #NOTE: Dimension += 1 do thêm một chiều của chỉ số gauss 
    # population = add_coefficient_gauss(population) 

    skill_factor = skill_factor_best_task(population, tasks)

    factorial_cost = cal_factor_cost(population, tasks, skill_factor)

    scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)


    
    # Khởi tạo có phần giống với cái xác suất của nó . 
    p_matrix= np.ones(shape= (len(tasks), len(tasks)), dtype= float) / len(tasks)
    # Khởi tạo một mảng các memory cho các task 
    memory_task = [Memory_SaMTPSO(len(tasks)) for i in range(len(tasks))]
    # gauss_mutation(population, population, population, population, population, population)
    history_cost = [] 
    history_p_matrix = []
    history_p_matrix.append(p_matrix)

   
    
    de = [DE() for i in range(len(tasks))]
    block = np.zeros(shape= (len(tasks)), dtype= int)
    ti_le_DE_gauss=np.zeros(shape= (len(tasks)), dtype= float) + 0.5 
    # sbx = newSBX(len(tasks), nc = 2, gamma= 0.9, alpha= 6)
    # sbx.get_dim_uss(DIMENSIONS+1)
    ti_le_dung_de = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) - 0.2
    #NOTE: new gauss jam: init
    gauss_mu_jam  = [gauss_mutation_jam() for i in range(len(tasks))] 
    count_loop = 0; 
    while np.sum(evaluations) < MAXEVALS:
        count_loop += 1 
        dc_sinh_ra_de = [] 

        childs = []
        skill_factor_childs = []
        factorial_cost_childs = []
        # XEM CON SINH RA Ở VỊ TRÍ THỨ BAO NHIÊU ĐỂ TÍ XÓA
        index_child_each_tasks = []

        # TASK ĐƯỢC CHỌN LÀM PARTNER LÀ TASK NÀO TƯƠNG ỨNG VỚI MỖI CON 
        task_partner = [] 

        list_population = np.arange(len(population))
        np.random.shuffle(list_population) 
        index= len(population) 
        number_child_each_task = np.zeros(shape=(len(tasks)), dtype = int)
        delta = [] 
      

        for task in range(len(tasks)):
            # if block[task] != 0: 
            #     continue 
            # if len(history_cost) > 2 and  history_cost[len(history_cost) - 1][task].cost == 0.0: 
            #     continue 
            while(number_child_each_task[task] < current_size_population[task]):
                # random de chon task nao de giao phoi 
                task2 = None 
                # index_pa= int(np.random.choice(np.where(skill_factor == task)[0], size= 1))
                
                index_pa = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(1.0 * current_size_population[task]))[0]) & set(np.where(skill_factor == task)[0])))), size= (1)))
                index_pb = index_pa 
                task2 = np.random.choice(np.arange(len(tasks)), p= p_matrix[task])
                while index_pa == index_pb:  
                    if task == task2: 
                        index_pb = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.5 * current_size_population[task2]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                    else:
                        index_pb = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(1.0 * current_size_population[task2]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                    # index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size = 1))
                # while index_pb == index_pa : 
                #     index_for_pb= np.random.choice(np.where(skill_factor == task2)[0], size= (2,), replace= False)
                #     if factorial_cost[index_for_pb[0]] < factorial_cost[index_for_pb[1]]:
                #         index_pb = int(index_for_pb[0]) 
                #     else: 
                #         index_pb = int(index_for_pb[1]) 

                # if memory_task[task].isFocus == False: 
                # if True:
                #     task2 = np.random.choice(np.arange(len(tasks)), p= p_matrix[task]) 
                #     while index_pa == index_pb:
                #         index_pb = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(1.0 * current_size_population[task2]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                #         # index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size= 1))
                # else: 
                #     task2 = task 
                #     while index_pa == index_pb:
                #         # index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size= 1))
                #         index_pb = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.25 * current_size_population[task]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                
                # CHỌN CHA MẸ # TRONG BÀI BÁO LÀ CHỌN CON KHỎE NHẤT. 
                # CROSSOVER 
                skf_oa = skf_ob= task
                # oa, ob = sbx(population[index_pa], population[index_pb], (task, task2))
                if task == task2: 
                    oa, ob = sbx_crossover(population[index_pa], population[index_pb], swap= True)
                else: 
                    oa, ob = sbx_crossover(population[index_pa], population[index_pb], swap= False)

                # oa = gauss_base_population(population[np.where(skill_factor == task)[0]], oa) 
                # ob = gauss_base_population(population[np.where(skill_factor == task)[0]], ob) 
                # #NOTE: new gauss jam: mutation 
                # if np.random.rand() < 0.05 and task == task2: 
                #     oa, _ = gauss_mu_jam[skf_oa].mutation(oa, population[np.where(skill_factor == skf_oa)[0]], skf_oa)
                #     ob, _ = gauss_mu_jam[skf_ob].mutation(ob, population[np.where(skill_factor == skf_ob)[0]], skf_ob)
                    

            
                fcost_oa = tasks[skf_oa].func(oa)
                fcost_ob = tasks[skf_ob].func(ob) 

                # tinh phan tram cai thien 
                delta_oa = (factorial_cost[index_pa] - fcost_oa) / (factorial_cost[index_pa] + 1e-10)
                delta_ob = (factorial_cost[index_pa] - fcost_ob) / (factorial_cost[index_pa] + 1e-10) 

                delta.append(delta_oa)
                delta.append(delta_ob) 

                skill_factor_childs.append(skf_oa) 
                skill_factor_childs.append(skf_ob) 

                factorial_cost_childs.append(fcost_oa) 
                factorial_cost_childs.append(fcost_ob) 

                childs.append(oa) 
                childs.append(ob) 


                
                index_child_each_tasks.append(index)
                index_child_each_tasks.append(index + 1) 

                task_partner.append(task2) 
                task_partner.append(task2) 

                number_child_each_task[task] += 2 
                evaluations[task] += 2
                index += 2 
    



        if lsa is True:
            current_size_population = Linear_population_size_reduction(
                evaluations,
                current_size_population,
                maxEvals,
                len(tasks),
                initial_size_population,
                min_size_population,
            )
     
        population = np.concatenate([population, np.array(childs)])
        skill_factor = np.concatenate([skill_factor, np.array(skill_factor_childs)])
        factorial_cost = np.concatenate(
            [factorial_cost, np.array(factorial_cost_childs)]
        )
        scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

        # Cập nhật index của bọn child cho mỗi tác vụ 
        # Check
        assert len(population) == len(skill_factor)
        assert len(population) == len(factorial_cost)

        #
        delete_index = [[] for i in range(len(tasks))]
        choose_index = [] 
        index_child_success = [] 
        index_child_fail = [] 
        delta_improvment = [] 
        delta_decrease = [] 
        nb_choosed_each_tasks = np.zeros(shape= (len(tasks),), dtype= int)
        for ind in range(len(population)):
      
            if(scalar_fitness[ind]) < 1.0 / (current_size_population[skill_factor[ind]] * ti_le_giu_lai) :
                delete_index[skill_factor[ind]].append(ind) 
            else: 
                choose_index.append(ind)
                nb_choosed_each_tasks[skill_factor[ind]] += 1 
               
            if ind >= index_child_each_tasks[0] : 
                task1 = skill_factor[ind]
                task2 = task_partner[ind - index_child_each_tasks[0]]

                if scalar_fitness[ind] < 1.0 / current_size_population[skill_factor[ind]]: 
                    memory_task[task1].update_history_memory(task2, 1, success= False)
                else: 
                    index_child_success.append(ind- index_child_each_tasks[0])
                    memory_task[task1].update_history_memory(task2, 1, success= True) 


        if ti_le_giu_lai < 1.0 : 
            for i in range(len(tasks)):
            # while nb_choosed_each_tasks[i] < current_size_population[i]:
                    # chọn random từ đống kia 1 vài con 
                choose_index = np.concatenate([np.array(choose_index),np.random.choice(delete_index[i], size= int(current_size_population[i] - nb_choosed_each_tasks[i]), replace = False)])
                    
        # Tính toán lại ma trận prob 
        for task in range(len(tasks)):
            p = np.copy(memory_task[task].compute_prob())
            assert p_matrix[task].shape == p.shape
            p_matrix[task] = p_matrix[task] * 0.8 + p * 0.2
            # p_matrix[task] = p 




        #: UPDATE QUẦN THỂ
        np.random.shuffle(choose_index)
        population = population[choose_index]
        scalar_fitness = scalar_fitness[choose_index]
        skill_factor = skill_factor[choose_index]
        factorial_cost = factorial_cost[choose_index]

        assert len(population) == np.sum(current_size_population)

        #NOTE: UPDATE SBX CODE KIEN 
        # sbx.update(index_child_success)
        # sbx.update_success_fail(index_child_success, delta_improvment, index_child_fail, delta_decrease, c=2)

        index_population_tasks = [[] for i in range(len(tasks))]
        for ind in range(len(population)): 
            index_population_tasks[skill_factor[ind]].append(ind) 
        

      
        

        xsuat_lay = 0
        # ANCHOR: learn phase DE
        for subpop in range(len(tasks)):
            # if gauss_mu_jam[subpop].is_jam:
            #     continue 
            count_mutation = 0; 
            count_de = 0; 
            danh_gia_DE = 0 
            danh_gia_Gauss= 0 
            
            # for ind in np.random.shuffle(index_population_tasks[subpop])[:int(len(index_child_each_tasks[subpop]))]:
            max_f = np.max(factorial_cost[np.array(list((set(np.where(scalar_fitness >= 1/(ti_le_giu_lai * current_size_population[subpop]))[0]) & set(np.where(skill_factor == subpop)[0]))))])
            min_f = np.min(factorial_cost[np.array(list((set(np.where(scalar_fitness >= 1/(ti_le_giu_lai * current_size_population[subpop]))[0]) & set(np.where(skill_factor == subpop)[0]))))])
            # min_f = np.min(factorial_cost[index_population_tasks[subpop]])
            np.random.shuffle(index_population_tasks[subpop])
       
            for ind in index_population_tasks[subpop][:int(len(index_population_tasks[subpop])/2)]:
                # if np.random.uniform() <ti_le_DE_gauss[subpop]:
                if np.random.uniform() < 1.1:

                    pbest = pr1 = pr2 = -1 
                   
                    while pbest == pr1 or pr1 == pr2: 
                        pbest= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.1 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
                        pr1= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(1.0 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
                        pr2= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(1.0 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
              
                    # pr1 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
                    # pr2 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
                    # pr1 = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.5 * current_size_population[task2]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))

                    new_ind = de[skill_factor[ind]].DE_cross(population[ind], population[pbest], population[pr1], population[pr2])

                    new_fcost = tasks[skill_factor[ind]].func(new_ind) 
                    delta_fcost = factorial_cost[ind] - new_fcost
                    evaluations[skill_factor[ind]] += 1
                    # xsuat_lay =  (1/mutation[skill_factor[ind]].tiso) * np.exp(delta_fcost /(max_f - min_f))
                    # if delta_fcost > 0 : 
                    danh_gia_DE += 1
                    # if (delta_fcost >=0 and factorial_cost[ind] > 0) or  (min_f > 0 and scalar_fitness[ind] < 1/(0.5* current_size_population[skill_factor[ind]]) and np.random.rand() < min_f / new_fcost ): 
                    if delta_fcost > 0 : 
                        if delta_fcost != 0:
                            count_de += 1       
                            de[skill_factor[ind]].update(delta_fcost) 
                        population[ind] = new_ind 
                        factorial_cost[ind]= factorial_cost[ind] - delta_fcost
                    
                else: 
                    (new_ind, std) = gauss_mu_jam[skill_factor[ind]].mutation(population[ind], population[np.where(skill_factor == subpop)[0]])
                    danh_gia_Gauss += 1
                    new_fcost = tasks[skill_factor[ind]].func(new_ind) 
                    delta_fcost = factorial_cost[ind] - new_fcost
                    evaluations[skill_factor[ind]] += 1
                    # xsuat_lay =  (1/mutation[skill_factor[ind]].tiso) * np.exp(delta_fcost /(max_f - min_f))
                    # if (delta_fcost >=0 and factorial_cost[ind] > 0) or (min_f > 0 and scalar_fitness[ind] < 1/(0.5* current_size_population[skill_factor[ind]]) and np.random.rand() < min_f / new_fcost): 
                    if delta_fcost > 0 :
                        # de[skill_factor[ind]].update(delta_fcost) 
                        if delta_fcost > 0 and factorial_cost[ind] > 0:
                            count_mutation += 1 
                            pass
                        # gauss_mu_jam[skill_factor[ind]].update()
                        # count_mutation += delta_fcost / factorial_cost[ind]
                        population[ind] = new_ind 
                        factorial_cost[ind]= new_fcost

            if danh_gia_DE> 0 and danh_gia_Gauss > 0:
                a = count_de / danh_gia_DE
                b= count_mutation/ danh_gia_Gauss
                if a== b and a == 0: 
                    ti_le_DE_gauss[subpop] -= (ti_le_DE_gauss[subpop] - 0.5) * 0.2
                else: 
                    x= np.max([a / (a + b), 0.2]) 
                    x= np.min([x, 0.8]) 
                    ti_le_DE_gauss[subpop] = ti_le_DE_gauss[subpop]* 0.5+ x * 0.5 
                    
            
        # ANCHOR: UPDATE DE :))
        for d in de: 
            d.reset()




        if int(evaluations[0] / 100) > len(history_cost):
            ti_le_giu_lai = (0.5 - 0.9) * np.sum(evaluations) / np.sum(maxEvals) + 0.9 
            
            results = optimize_result(population, skill_factor, factorial_cost, tasks)
            history_cost.append(results)

            #NOTE: new gauss jam: update 
            end = len(history_cost) -1 
            for i in range(len(tasks)):
                gauss_mu_jam[i].update_scale(history_cost[end][i].cost) 


            history_p_matrix.append(np.copy(p_matrix))
            sys.stdout.flush()
            sys.stdout.write("\r")
            from time import sleep
            
            sys.stdout.write(
                "Epoch {}, [%-20s] %3d%% ,pop_size: {},count_de: {},  func_val: {}, gauss: {}".format(
                    int(evaluations[0] / 100) + 1,
                    len(population),
                    [evaluations[i] for i in range(len(tasks))],
                    # [gauss_mu_jam[i].is_jam for i in range(len(tasks))],
                    # [gauss_mu_jam[i].curr_diversity for i in range(len(tasks))],
                    # [i for i in ti_le_dung_de],
                    # count_mutation,
                    # [memory_task[9].success_history[i][memory_task[9].next_position_update-1] for i in range(len(tasks))],
                    # [mutation[i].jam for i in range(NUMBER_TASKS)],
                    # [p_matrix[6][i] for i in range(NUMBER_TASKS)],
                    [results[i].cost for i in range(len(tasks))],
                    [ti_le_DE_gauss[i] for i in range(len(tasks))],
                )
                % (
                    "=" * np.int((np.sum(evaluations) + 1) // (MAXEVALS // 20)) + ">",
                    (np.sum(evaluations) + 1) * 100 // MAXEVALS,
                )
            )
            sys.stdout.flush()
    print("")
    print(count_loop)
    for i in range(len(tasks)):
        print(tasks[i].name, ": ", results[i].cost)
    return history_cost, history_p_matrix, population, skill_factor, scalar_fitness


class Memory_SaMTPSO:
    def __init__(self, number_tasks, LP=10) -> None:
        self.success_history = np.zeros(shape=(number_tasks, LP), dtype=float)
        self.fail_history = np.zeros(shape=(number_tasks, LP), dtype=float)
        self.next_position_update = 0  # index cột đang được update
        self.isFocus = False
        self.epsilon = 0.0001
        self.pb = 0.005
        self.LP = LP
        self.duytri = 10
        self.number_tasks = number_tasks

    def update_history_memory(self, task_partner, delta, success=True, end_generation=False) -> None:
        if success:
            self.success_history[task_partner][self.next_position_update] += delta
        else:
            self.fail_history[task_partner][self.next_position_update] += delta

        # Nếu hết một thế hệ -> tăng vị trí cập nhật success và fail memory lên một ->
        # if end_generation:
            # Nếu trong LP thế hệ gần nhất mà không cải thiện được thì -> tập trung intra

    def compute_prob(self) -> np.ndarray:
        sum = np.clip(np.sum(self.success_history, axis=1) +
                      np.sum(self.fail_history, axis=1), 0, 100000)
        sum_sucess = np.clip(np.sum(self.success_history, axis=1), 0, 10000)

        SRtk = sum_sucess / (sum + self.epsilon) + self.pb
        p = SRtk / np.sum(SRtk)
        # FIXME: LÀM SAO BIẾT ĐƯỢC BAO NHIÊU LÀ ĐỦ ĐỂ CHO FOCUS ?
        if np.sum(self.success_history[:, self.next_position_update]) == 0 and self.duytri <= 0:
            self.isFocus = not self.isFocus
            self.duytri = 10

        if np.sum(self.success_history[:, self.next_position_update]) != 0 and self.isFocus is True and self.duytri <= 0:
            self.isFocus = False
            self.duytri = 10
        self.duytri -= 1
        self.next_position_update = (self.next_position_update + 1) % self.LP
        self.success_history[:, self.next_position_update] = 0
        self.fail_history[:, self.next_position_update] = 0
        return p

    # def compute_prob(self)->np.ndarray:

class DE: 
    def __init__(self) -> None:
        # DE
        self.Mcr = np.zeros(shape= (30), dtype= float) + 0.5
        self.Mf = np.zeros(shape= (30), dtype= float) + 0.5
        self.index_update = 0 
        # cr and r temp 
        self.Scr = [] 
        self.Sf = [] 
        self.w = [] 

        self.rate_improve = 0 
        self.func_eval = 0 

        self.name = "DE"

        # luu tam thoi 
        self.cr_tmp = 0 
        self.f_tmp = 0 

    def DE_cross(self,p, pbest, pr1, pr2)->np.array: 
        '''
        pbest: chon random tu 10% on top 
        pr1, pr2: chon random tu quan the voi task tuong ung thoi 
        '''
        D = len(pbest) - 1
        jrand = np.random.choice(np.arange(D), size = 1) 
        D = len(pbest)
        y = np.zeros_like(pbest) 
        k = np.random.choice(np.arange(len(self.Mcr)), size= 1)
        cr = np.random.normal(loc= self.Mcr[k], scale = 0.1) 
        if cr > 1:
            cr = 1 
        if cr <=0: 
            cr = 0
        # cr = scipy.stats.cauchy.rvs(loc= self.Mcr[k], scale= 0.1)
        F = 0
        while F <= 0: 
            F = scipy.stats.cauchy.rvs(loc= self.Mf[k], scale= 0.1) 
        if F > 1: 
            F = 1 
        # for i in range(D):
        #     if np.random.uniform() < cr or i == jrand: 
        #         y[i] = pbest[i] + F * (pr1[i] - pr2[i])
        #     else: 
        #         y[i] =  p[i] 
        u = np.random.uniform(size= D) 
        y[u < cr] = pbest[u< cr] + F*(pr1[u < cr] - pr2[u < cr])
        y[u >= cr] = p[u >= cr]
        y[jrand] =  pbest[jrand] + F * (pr1[jrand] - pr2[jrand])
        self.cr_tmp = cr 
        self.f_tmp = F 
        y = np.clip(y, 0,1)
        return y 
    def performance(self)->float:
        if self.func_eval == 0:
            return 1 
        else: 
            return self.rate_improve / self.func_eval 
    def update(self, delta_fcost):
        if delta_fcost > 0: 
            self.rate_improve += delta_fcost 

            self.Scr.append(float(self.cr_tmp))
            self.Sf.append(float(self.f_tmp))
            self.w.append(float(delta_fcost))
        self.func_eval += 1 
    
    def reset(self): 
        sum_w = np.sum(self.w) 
        new_cr = 0 
        new_f = 0 
        new_index = (self.index_update +1) % len(self.Mcr) 
        if len(self.Scr) > 0:
            # for i in range(len(self.Scr)): 
            #     new_cr += self.Scr[i] * self.w[i] / sum_w 
            #     new_f += self.Sf[i] * self.w[i] / sum_w 
            new_cr = np.sum(np.array(self.Scr) * (np.array(self.w) / sum_w) )
            new_f = (np.sum(np.array(self.w) * np.array(self.Sf) ** 2)) / (np.sum(np.array(self.w )*np.array(self.Sf)))
            self.Mcr[new_index] = new_cr 
            self.Mf[new_index] = new_f 
            
        else: 
            self.Mcr[new_index] = np.copy(self.Mcr[self.index_update])
            self.Mf[new_index] = np.copy(self.Mf[self.index_update])
        
        self.index_update = new_index 
        self.Scr.clear() 
        self.Sf.clear() 
        self.w.clear() 
        self.func_eval = 0 
        self.rate_improve = 0 

class gauss_mutation_jam:
    def __init__(self) -> None:
        self.is_jam = False
        self.scale = None

        self.cost_10_pre_gen = None
        self.curr_cost = None
        self.count_gen = 0

        self.max_log = np.zeros(shape=(DIMENSIONS, ), dtype=int) + 2
        self.min_log = 1

        self.count_mu_each_dim = np.zeros(shape=(DIMENSIONS,), dtype=int)
        self.diversity_begin: float = -1
        self.curr_diversity: float = -1

        self.count_mu_each_scale = np.zeros(shape=(DIMENSIONS, 30), dtype=int)

        self.his_scale_each_dim = [[] for i in range(DIMENSIONS,)]

        self.add = 0 

    def update_scale(self, best_cost):
        if self.cost_10_pre_gen is None and self.curr_cost is None:
            self.curr_cost = best_cost
            self.cost_10_pre_gen = best_cost
        else:
            self.curr_cost = best_cost
            self.count_gen += 1
        if self.count_gen > 10 and self.cost_10_pre_gen > 0:
            delta = (self.cost_10_pre_gen - self.curr_cost) / \
                self.cost_10_pre_gen
            if delta < 0.1:
                self.is_jam = True
            else:
                self.is_jam = False

            self.cost_10_pre_gen = best_cost
            self.count_gen = 0

    def mutation(self, ind, subpopulation=None, skf=0, need_diversity=False):
        D = DIMENSIONS

        p_for_dim = 1 / (self.count_mu_each_dim + 1)
        p_for_dim /= np.sum(p_for_dim)
        i = int(np.random.choice(np.arange(D), size=1, p=p_for_dim))

        # mean = np.mean(subpopulation[:, i])
        std = np.std(subpopulation[:, i])
        e = -1

        if std != 0:
            log = np.log10(std)
            log = int(np.abs(log))+1
            if log+1 > self.max_log[i]:
                self.max_log[i] = log + 1
                # NOTE
                # self.count_mu_each_dim = np.zeros_like(self.count_mu_each_dim) + 1
                self.count_mu_each_scale[i] = np.zeros_like(
                    self.count_mu_each_scale[i]) + 1

        rand = np.random.rand()
        rand_scale = 1 / (self.count_mu_each_scale[i][1:self.max_log[i]] + 1)
        p = np.array(rand_scale / np.sum(rand_scale))

        if std == 0:
            e = int(np.random.choice(
                np.arange(self.max_log[i])[1:], size=1, p=p))
            e = -e
            self.scale = 10 ** e

        if std != 0:
            e = np.random.choice(np.arange(self.max_log[i])[1:], size=1, p=p)
            e = int(e)
            e = -1 * e
            self.scale = 10 ** e

        # if rand <= 0.1 and self.is_jam is False
        self.dim_temp = i
        if False:
            pa = np.random.choice(
                np.arange(len(subpopulation)), size=(2,), replace=False)
            scale = np.abs(subpopulation[pa[0]][i] - subpopulation[pa[1]][i])
            t = ind[i] + np.random.normal(loc=0, scale=scale)
        else:
            self.count_mu_each_dim[i] += 1
            self.count_mu_each_scale[i][np.abs(e)] += 1
            self.his_scale_each_dim[i].append(np.abs(e))
            self.add = np.random.normal(loc=0, scale=self.scale)
            t = ind[i] + self.add
        if t > 1:
            self.add = np.random.rand() * (1 - ind[i])
            t = ind[i] + self.add
        elif t < 0:
            self.add = np.random.rand() * np.abs(ind[i]) - ind[i]
            t = ind[i] + self.add 

        ind[i] = t

        return ind, std
    def update(self): 
        if self.add != 0 : 
            e = 0 
            while np.log10(np.abs(self.add)) + e < 1:
                e += 1 
            self.count_mu_each_scale[self.dim_temp][np.abs(e-1)] = max(1, self.count_mu_each_scale[self.dim_temp][np.abs(e-1)] - 1)
            
                 



def sbx_crossover(p1, p2, nc = 2, swap = True):
    SBXDI = nc
    D = p1.shape[0]
    cf = np.empty([D])
    u = np.random.rand(D)        

    cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (SBXDI + 1)))
    cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (SBXDI + 1)))

    c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
    c2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)
    c1 = np.clip(c1, 0, 1)
    c2 = np.clip(c2, 0, 1)
    if swap is True: 
        c1, c2 = variable_swap(c1,c2, 0.5)

    return c1, c2

def variable_swap(p1, p2, probswap):
    D = p1.shape[0]
    swap_indicator = np.random.rand(D) <= probswap
    c1, c2 = p1.copy(), p2.copy()
    c1[np.where(swap_indicator)] = p2[np.where(swap_indicator)]
    c2[np.where(swap_indicator)] = p1[np.where(swap_indicator)]
    return c1, c2

def Linear_population_size_reduction(
    evaluations, current_size_population, maxEvaluations, number_tasks, maxSize, minSize
):
    for task in range(number_tasks):
        new_size = (minSize[task] - maxSize[task]) / maxEvaluations[task] * evaluations[
            task
        ] + maxSize[task]
        if new_size < current_size_population[task] and new_size >= minSize[task]:
            current_size_population[task] = new_size
    return current_size_population


def optimize_result(population, skill_factor, factorial_cost, tasks):
    class result:
        def __init__(self, cost=1e10, task=-1):
            self.cost = cost
            self.task = task
            self.ind = None

    results = [result(task=i) for i in range(np.max(skill_factor) + 1)]

    for i in range(len(population)):
        if results[skill_factor[i]].cost > factorial_cost[i]:
            results[skill_factor[i]].cost = factorial_cost[i]
            results[skill_factor[i]].ind = population[i] 
    # for result in results:
    #     print("tasks: {} | cost: {} ".format(result.task, result.cost))

    return results
