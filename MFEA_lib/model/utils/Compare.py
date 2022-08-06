from .. import AbstractModel
from typing import List, Tuple
import numpy as np 
import os
from .Run import *
import pandas as pd 
import matplotlib.pyplot as plt 



class CompareResultBenchmark:
    '''
    Show result multibenchmark
    ''' 
    def __init__(self, path_folder: str = None, ls_benchmark: list = [], ls_name_algo = [], load_folder = True) -> None:
        self.path_folder = path_folder 
        self.ls_benchmark = ls_benchmark
        self.ls_name_algo = ls_name_algo 
        if load_folder is True: 
            self.load_folder() 
        pass

    def load_folder(self): 
        ls_algorithms = os.listdir(self.path_folder) 
        if len(self.ls_name_algo) == 0: 
            self.ls_name_algo = ls_algorithms.copy() 
        for idx, name in enumerate(self.ls_name_algo): 
            print(f"({idx} : {name})")
        
    def show_compare_detail(self, min_value= 0, round= 100, idx_main_algo= 0, idx_gener_compare = -1, total_generation = 1000):
        # Step1: read folder 
        algo_ls_model = np.zeros(shape=(len(self.ls_name_algo), len(self.ls_benchmark))).tolist() 
        ls_algorithms = os.listdir(self.path_folder)
        # ls_benchmark_each_algo = [[]]
        count_benchmark = np.zeros(shape=(len(self.ls_benchmark)), dtype= int)

        if len(self.ls_name_algo) == 0: 
            self.ls_name_algo = ls_algorithms.copy()  
        assert len(self.ls_name_algo) == len(ls_algorithms)
        # Step2: Create ls model of each benchmark
        for idx_algo, algorithm in enumerate(ls_algorithms): 
            path_model = os.path.join(self.path_folder, algorithm)
            ls_models = os.listdir(path_model) 
            for model_name in ls_models: 
                idx_benchmark = (model_name.split(".")[0]).split("_")[-1] 
                idx_benchmark = int(idx_benchmark)-1
                # ls_benchmark_each_algo[idx_algo].append(idx_benchmark)
                # try:
                count_benchmark[idx_benchmark] += 1
                model = loadModel(os.path.join(path_model, model_name), self.ls_benchmark[int(idx_benchmark)]) 
                try:
                    model.set_attribute()
                except:
                    # print('koco')
                    pass
                algo_ls_model[idx_algo][idx_benchmark] = model 
                # except: 
                #     print(f"Cannot load Model: {os.path.join(path_model, model_name)}")    
                #     return
        
        # swap 
        algo_ls_model[0], algo_ls_model[idx_main_algo] = algo_ls_model[idx_main_algo], algo_ls_model[0] 
        # self.ls_name_algo[0], self.ls_name_algo[idx_main_algo] = self.ls_name_algo[idx_main_algo], self.ls_name_algo[0] 


        # Step3: use compare model for each model in benchmark  
        for benchmark in range(len(self.ls_benchmark)): 
            print("Benchmark: ", benchmark + 1)
            if count_benchmark[benchmark] == 0: 
                continue 
            try: 
                # compare = CompareModel([algo_ls_model[i][benchmark] for i in range(len(self.ls_name_algo))])
                # print(compare.detail_compare_result(min_value= min_value, round = round))
                name_row = [str("Tasks") + str(i+1) for i in range(len(self.ls_benchmark[0]))]
                name_col = np.copy(self.ls_name_algo)

                name_col[0], name_col[idx_main_algo] = name_col[idx_main_algo], name_col[0]

                ls_models = [algo_ls_model[i][benchmark] for i in range(len(self.ls_name_algo))]
                shape_his = None 
                for model in ls_models: 
                    if model  != 0 : 
                        shape_his = model.history_cost[-1].shape
                if shape_his == None: 
                    continue 

                data = []
                for model in ls_models:
                    if model == 0 : 
                        data.append(np.zeros(shape= shape_his) + 1e20)
                        continue
                    idx_compare = -1
                    if idx_gener_compare == -1 or idx_gener_compare == total_generation: 
                        idx_compare = -1 
                    else: 
                        idx_compare = int(idx_gener_compare/total_generation* len(model.history_cost) )
                    data.append(model.history_cost[idx_compare]) 
                
                data = np.array(data).T 
                data = np.round(data, round) 

                end_data = pd.DataFrame(data).astype(str) 

                result_compare = np.zeros(shape=(len(name_col)), dtype=int).tolist() 

                for task in range(len(name_row)): 
                    argmin= np.argmin(data[task])
                    min_value_ = max(data[task][argmin], min_value) 

                    for col in range(len(name_col)): 
                        if data[task][col] <= min_value_: 
                            result_compare[col] += 1 
                            end_data.iloc[task][col] = str("(+)") + end_data.iloc[task][col]
                
                for col in range(len(name_col)):
                    result_compare[col] = str(result_compare[col]) + "/" + str(len(name_row))
                
                result_compare = pd.DataFrame([result_compare], index=["Compare"], columns= name_col) 
                end_data.columns = name_col 
                end_data.index = name_row 

                pd.set_option('display.expand_frame_repr', False)
                end_data = pd.concat([end_data, result_compare]) 
                print(end_data)
                print()
            except: 
                print(f"Cannot compare benchmark {benchmark+1}")
                pass 
        pass

    def summarizing_compare_result_v2(self, idx_main_algo=0, min_value=0, combine=True):
        nb_task = len(self.ls_benchmark[0])
        list_algo = os.listdir(self.path_folder)
        print(list_algo)
        benchmarks = [name_ben.split(
            "_")[-1].split(".")[0] for name_ben in os.listdir(os.path.join(self.path_folder, list_algo[0]))]
        ls_model_cost = [np.zeros(
            shape=(len(benchmarks), nb_task)).tolist() for i in range(len(list_algo))]
        # print(ls_model_cost)

        ls_benhchmark = [] 
        for idx_algo in range(len(list_algo)):
            path_algo = os.path.join(self.path_folder, list_algo[idx_algo])
            # count_benchmark = 0

            for benchmark_mso in os.listdir(path_algo):
                count_benchmark = benchmark_mso.split(".")[0]
                count_benchmark = count_benchmark.split("_")[-1]
                count_benchmark = int(count_benchmark) - 1

                model = loadModel(os.path.join(
                    path_algo, benchmark_mso), self.ls_benchmark[count_benchmark])

                ls_model_cost[idx_algo][count_benchmark] = model.history_cost[-1]
                # count_benchmark += 1

        result_table = np.zeros(
            shape=(len(benchmarks), len(list_algo)-1, 3), dtype=int)
        name_row = []
        name_col = ["Better", "Equal", "Worse"]
        count_row = 0
        for idx_algo in range(len(list_algo)):
            if idx_main_algo != idx_algo:
                name_row.append(
                    list_algo[idx_main_algo] + " vs " + list_algo[idx_algo])
                for idx_benchmark in range(len(benchmarks)):
                    result = np.where(ls_model_cost[idx_main_algo][idx_benchmark] > min_value, ls_model_cost[idx_main_algo][idx_benchmark], min_value) \
                        - np.where(ls_model_cost[idx_algo][idx_benchmark] > min_value,
                                    ls_model_cost[idx_algo][idx_benchmark], min_value)

                    result_table[idx_benchmark][count_row][0] += len(
                        np.where(result < 0)[0])
                    result_table[idx_benchmark][count_row][1] += len(
                        np.where(result == 0)[0])
                    result_table[idx_benchmark][count_row][2] += len(
                        np.where(result > 0)[0])
                count_row += 1
        if combine is True:
            result_table = pd.DataFrame(
                np.sum(result_table, axis=0), columns=name_col, index=name_row)
        return result_table
    
    def summarizing_compare_result(self, idx_main_algo= 0, min_value= 0, combine = True, idx_gener_compare = -1, total_generation = 1000): 
        # Step1: read folder 
        algo_ls_model = np.zeros(shape=(len(self.ls_name_algo), len(self.ls_benchmark))).tolist() 
        ls_algorithms = os.listdir(self.path_folder)

        if len(self.ls_name_algo) == 0: 
            self.ls_name_algo = ls_algorithms.copy()  
        assert len(self.ls_name_algo) == len(ls_algorithms)
        # Step2: Create ls model of each benchmark
        self.ls_idx_benchmark = np.zeros(shape=(len(self.ls_benchmark))).tolist()  
        for idx_algo, algorithm in enumerate(ls_algorithms): 
            path_model = os.path.join(self.path_folder, algorithm) 
            ls_models = os.listdir(path_model) 
            for model_name in ls_models: 
                idx_benchmark = (model_name.split(".")[0]).split("_")[-1] 
                idx_benchmark = int(idx_benchmark)-1
                self.ls_idx_benchmark[idx_benchmark] += 1 
                # try:
                model = loadModel(os.path.join(path_model, model_name), self.ls_benchmark[int(idx_benchmark)]) 
                try:
                    model.set_attribute()
                except:
                    # print('koco')
                    pass
                algo_ls_model[idx_algo][idx_benchmark] = model 
                # except: 
                #     print(f"Cannot load Model: {os.path.join(path_model, model_name)}")    
                #     return
        
        # tìm xem thuật toán nào ko có bộ benchmark thì bỏ ko so sánh 
        name_row = [] 
        name_col =['Better', 'Equal', 'Worse']
        count_row = 0 
        result_table = np.zeros(shape=(len(self.ls_benchmark), len(ls_algorithms) -1, 3), dtype= int)
        for idx, name_algo in enumerate(ls_algorithms): 
            if idx != idx_main_algo: 
                name_row.append(ls_algorithms[idx_main_algo] + " vs " + name_algo)
                for idx_benchmark in range(len(self.ls_benchmark)):
                    if algo_ls_model[idx][idx_benchmark] == 0 or algo_ls_model[idx_main_algo][idx_benchmark] == 0 : 
                        continue 

                    idx_gener_compare_first = -1  
                    idx_gener_compare_second = -1  
                    if idx_gener_compare == total_generation or idx_gener_compare == -1: 
                        idx_gener_compare_first = -1
                        idx_gener_compare_second = -1
                    else: 
                      
                        idx_gener_compare_first = int(idx_gener_compare/total_generation * len(algo_ls_model[idx][idx_benchmark].history_cost))
                        idx_gener_compare_second = int(idx_gener_compare/total_generation * len(algo_ls_model[idx_main_algo][idx_benchmark].history_cost))
                    result = np.where(algo_ls_model[idx][idx_benchmark].history_cost[idx_gener_compare_first] > min_value, algo_ls_model[idx][idx_benchmark].history_cost[idx_gener_compare_first], min_value) - np.where(algo_ls_model[idx_main_algo][idx_benchmark].history_cost[idx_gener_compare_second] > min_value, algo_ls_model[idx_main_algo][idx_benchmark].history_cost[idx_gener_compare_second], min_value)
                    result_table[idx_benchmark][count_row][2] += len(np.where(result < 0)[0]) 
                    result_table[idx_benchmark][count_row][1] += len(np.where(result == 0)[0])
                    result_table[idx_benchmark][count_row][0] += len(np.where(result > 0)[0]) 
                
                count_row += 1 

        if combine is True :
            result_table= pd.DataFrame(np.sum(result_table, axis= 0), columns= name_col, index= name_row) 
        
        return result_table
                


class CompareModel():
    # TODO so sánh
    def __init__(self, models: List[AbstractModel.model], label: List[str] = None) -> None:
        self.models = models
        if label is None:
            label = [m.name for m in self.models]
        else:
            assert len(self.models) == len(label)
            for idx in range(len(label)):
                if label[idx] == Ellipsis:
                    label[idx] = self.models[idx].name

        self.label = label

    def render(self, shape: tuple = None, min_cost=0, nb_generations: int = None, step=1, figsize: Tuple[int, int] = None, dpi=200, yscale: str = None, re=False, label_shape=None, label_loc=None):
        assert np.all([len(self.models[0].tasks) == len(m.tasks)
                      for m in self.models])
        nb_tasks = len(self.models[0].tasks)
        for i in range(nb_tasks):
            assert np.all([self.models[0].tasks[i] == m.tasks[i]
                          for m in self.models])

        if label_shape is None:
            label_shape = (1, len(self.label))
        else:
            assert label_shape[0] * label_shape[1] >= len(self.label)

        if label_loc is None:
            label_loc = 'lower center'

        if shape is None:
            shape = (nb_tasks // 3 + np.sign(nb_tasks % 3), 3)
        else:
            assert shape[0] * shape[1] >= nb_tasks

        if nb_generations is None:
            nb_generations = min([len(m.history_cost) for m in self.models])
        else:
            nb_generations = min(nb_generations, min(
                [len(m.history_cost) for m in self.models]))

        if figsize is None:
            figsize = (shape[1] * 6, shape[0] * 5)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.suptitle("Compare Models\n", size=15)
        fig.set_facecolor("white")
        fig.subplots(shape[0], shape[1])

        if step >= 10:
            marker = 'o'
        else:
            marker = None

        for idx_task, task in enumerate(self.models[0].tasks):
            for idx_model, model in enumerate(self.models):
                fig.axes[idx_task].plot(
                    np.append(np.arange(0, nb_generations, step),
                              np.array([nb_generations - 1])),
                    np.where(
                        model.history_cost[np.append(np.arange(0, nb_generations, step), np.array([
                                                     nb_generations - 1])), idx_task] >= min_cost,
                        model.history_cost[np.append(np.arange(0, nb_generations, step), np.array([
                                                     nb_generations - 1])), idx_task],
                        0
                    ),
                    label=self.label[idx_model],
                    marker=marker,
                )
                # plt.legend()
                if yscale is not None:
                    fig.axes[idx_task].set_yscale(yscale)
            fig.axes[idx_task].set_title(task.name)
            fig.axes[idx_task].set_xlabel("Generations")
            fig.axes[idx_task].set_ylabel("Cost")

        for idx_blank_fig in range(idx_task + 1, shape[0] * shape[1]):
            fig.delaxes(fig.axes[idx_task + 1])

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc=label_loc, ncol=label_shape[1])
        plt.show()
        if re:
            return fig

    def summarizing_compare_result(self, path=None, idx_main_algo=0, min_value=0, combine=True, nb_task=50, ls_benchmark=None):
        #
        if path is None:
            result_table = np.zeros(shape=(len(self.label)-1, 3), dtype=int)
            name_row = []
            name_column = ["Better", "Equal", "Worse"]
            count_row = 0
            for idx_algo in range(len(self.label)):
                if idx_algo != idx_main_algo:
                    name_row.append(
                        str(self.label[idx_main_algo] + " vs " + self.label[idx_algo]))
                    result = np.where(self.models[idx_main_algo].history_cost[-1] > min_value, self.models[idx_main_algo].history_cost[-1], min_value)\
                        - np.where(self.models[idx_algo].history_cost[-1] > min_value,
                                   self.models[idx_algo].history_cost[-1], min_value)
                    # Better
                    result_table[count_row][0] += len(np.where(result < 0)[0])
                    # Equal
                    result_table[count_row][1] += len(np.where(result == 0)[0])
                    # Worse
                    result_table[count_row][2] += len(np.where(result > 0)[0])
                    count_row += 1

            result_table = pd.DataFrame(
                result_table, columns=name_column, index=name_row)
        else:
            list_algo = os.listdir(path)
            print(list_algo)
            benchmarks = [name_ben.split(
                "_")[-1].split(".")[0] for name_ben in os.listdir(os.path.join(path, list_algo[0]))]
            ls_model_cost = [np.zeros(
                shape=(len(benchmarks), nb_task)).tolist() for i in range(len(list_algo))]
            # print(ls_model_cost)
            for idx_algo in range(len(list_algo)):
                path_algo = os.path.join(path, list_algo[idx_algo])
                # count_benchmark = 0

                for benchmark_mso in os.listdir(path_algo):
                    count_benchmark = benchmark_mso.split(".")[0]
                    count_benchmark = count_benchmark.split("_")[-1]
                    count_benchmark = int(count_benchmark) - 1

                    model = loadModel(os.path.join(
                        path_algo, benchmark_mso), ls_benchmark[count_benchmark])

                    ls_model_cost[idx_algo][count_benchmark] = model.history_cost[-1]
                    # count_benchmark += 1

            result_table = np.zeros(
                shape=(len(benchmarks), len(list_algo)-1, 3), dtype=int)
            name_row = []
            name_col = ["Better", "Equal", "Worse"]
            count_row = 0
            for idx_algo in range(len(list_algo)):
                if idx_main_algo != idx_algo:
                    name_row.append(
                        list_algo[idx_main_algo] + " vs " + list_algo[idx_algo])
                    for idx_benchmark in range(len(benchmarks)):
                        result = np.where(ls_model_cost[idx_main_algo][idx_benchmark] > min_value, ls_model_cost[idx_main_algo][idx_benchmark], min_value) \
                            - np.where(ls_model_cost[idx_algo][idx_benchmark] > min_value,
                                       ls_model_cost[idx_algo][idx_benchmark], min_value)

                        result_table[idx_benchmark][count_row][0] += len(
                            np.where(result < 0)[0])
                        result_table[idx_benchmark][count_row][1] += len(
                            np.where(result == 0)[0])
                        result_table[idx_benchmark][count_row][2] += len(
                            np.where(result > 0)[0])
                    count_row += 1
            if combine is True:
                result_table = pd.DataFrame(
                    np.sum(result_table, axis=0), columns=name_col, index=name_row)
        return result_table

    def detail_compare_result(self, min_value=0, round = 100):
        name_row = [str("Task" + str(i + 1))
                    for i in range(len(self.models[0].tasks))]
        name_col = self.label
        data = []
        for model in self.models:
            data.append(model.history_cost[-1])

        data = np.array(data).T
        data = np.round(data, round)
        pre_data = pd.DataFrame(data)
        end_data = pd.DataFrame(data).astype(str)

        result_compare = np.zeros(shape=(len(name_col)), dtype=int).tolist()
        for task in range(len(name_row)):
            argmin = np.argmin(data[task])
            min_value_ = max(data[task][argmin], min_value)
            # for col in range(len(name_col)):
            #     if data[task][col] == data[task][argmin]:
            #         result_compare[col] += 1
            #         end_data.iloc[task][col]= str("(≈)") + pre_data.iloc[task][col].astype(str)
            #     elif data[task][col] > data[task][argmin]:
            #         end_data.iloc[task][col]= str("(-)") + pre_data.iloc[task][col].astype(str)
            #     else:

            for col in range(len(name_col)):
                if data[task][col] <= min_value_:
                    result_compare[col] += 1
                    end_data.iloc[task][col] = str(
                        "(+)") + end_data.iloc[task][col]

        for col in range(len(name_col)):
            result_compare[col] = str(
                result_compare[col]) + "/" + str(len(name_row))

        result_compare = pd.DataFrame([result_compare], index=[
                                      "Compare"], columns=name_col)
        end_data.columns = name_col
        end_data.index = name_row
        end_data = pd.concat([end_data, result_compare])

        # assert data.shape == (len(name_row), len(name_col))

        return end_data
