from operator import index
import pickle
import matplotlib.pyplot as plt
import numpy as np
from ..tasks.function import AbstractTask
from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from pathlib import Path
import traceback
import os
import pandas as pd


def get_model_name(model: AbstractModel.model):
    fullname = model.__module__
    index = None
    for i in range(len(fullname) - 1, -1, -1):
        if fullname[i] == '.':
            index = i
            break
    if index is None:
        return fullname
    return fullname[index + 1:]


class MultiTimeModel:
    def __init__(self, model: AbstractModel, list_attri_avg: list = None,  name=None) -> None:

        self.model = model.model

        if name is None:
            self.name = model.__name__
        else:
            self.name = name

        if list_attri_avg is None:
            self.list_attri_avg = None
        else:
            self.list_attri_avg = list_attri_avg

        self.ls_model: list[AbstractModel.model] = []
        self.ls_seed: list[int] = []
        self.total_time = 0

        # add inherit
        cls = self.__class__
        self.__class__ = cls.__class__(cls.__name__, (cls, self.model), {})

        # status of model run
        # self.status = 'NotRun' | 'Running' | 'Done'
        self.status = 'NotRun'

    def set_data(self, history_cost: np.ndarray):
        self.status = 'Done'
        self.history_cost = history_cost
        print('Set complete!')

    def set_attribute(self):
        # print avg
        if self.list_attri_avg is None:
            self.list_attri_avg = self.ls_model[0].ls_attr_avg
        for i in range(len(self.list_attri_avg)):
            try:
                result = [model.__getattribute__(
                    self.list_attri_avg[i]) for model in self.ls_model]
            except:
                print("cannot get attribute {}".format(self.list_attri_avg[i]))
                continue
            try:
                result = np.array(result)
                result = np.average(result, axis=0)
                self.__setattr__(self.list_attri_avg[i], result)
            except:
                print(f'can not convert {self.list_attri_avg[i]} to np.array')
                continue

    def print_result(self, print_attr = [], print_time = True, print_name= True):
        # print time
        seconds = self.total_time
        minutes = seconds // 60
        seconds = seconds - minutes * 60
        if print_time: 
            print("total time: %02dm %.02fs" % (minutes, seconds))

        # print avg
        if self.list_attri_avg is None:
            self.list_attri_avg = self.ls_model[0].ls_attr_avg

        if len(print_attr) ==0 : 
            print_attr = self.list_attri_avg 
        
        for i in range(len(print_attr)):
            try:
                result = self.__getattribute__(print_attr[i])[-1]
                if print_name: 
                    print(f"{print_attr[i]} avg: ")
                np.set_printoptions(
                    formatter={'float': lambda x: format(x, '.2E')})
                print(result)
            except:
                try:
                    result = [model.__getattribute__(
                        print_attr[i]) for model in self.ls_model]
                    result = np.array(result)
                    result = np.sum(result, axis=0) / len(self.ls_model)
                    if print_name: 
                        print(f"{print_attr[i]} avg: ")
                    np.set_printoptions(
                        formatter={'float': lambda x: format(x, '.2E')})
                    print(result)
                except:
                    print(
                        f'can not convert {print_attr[i]} to np.array')

    def compile(self, **kwargs):
        self.compile_kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)

    def run(self, nb_run: int = None, save_path: str = "./RESULTS/result/model.mso", seed_arr: list = None, random_seed: bool = False, replace_folder = True):
        print('Checking ...', end='\r')
        # folder
        idx = len(save_path) 
        while save_path[idx-1] != "/": 
            idx -= 1 

        if os.path.isdir(save_path[:idx]) is True:
            if replace_folder is True:
                pass
            else:
                raise Exception("Folder is existed")
        else:
            os.makedirs(save_path[:idx])

        if self.status == 'NotRun':
            if nb_run is None:
                self.nb_run = 1
            else:
                self.nb_run = nb_run

            if save_path is None:
                save_path = get_model_name(self.model) + '.mso'

            if seed_arr is not None:
                assert len(seed_arr) == nb_run
            elif random_seed:
                seed_arr = np.random.randint(
                    nb_run * 100, size=(nb_run, )).tolist()
            else:
                seed_arr = np.arange(nb_run).tolist()

            self.ls_seed = seed_arr
            index_start = 0
        elif self.status == 'Running':
            if nb_run is not None:
                assert self.nb_run == nb_run

            if save_path is None:
                save_path = get_model_name(self.model) + '.mso'

            if seed_arr is not None:
                assert np.all(
                    seed_arr == self.ls_seed), '`seed_arr` is not like `ls_seed`'

            index_start = len(self.ls_model)
        elif self.status == 'Done':
            print('Model has already completed before.')
            return
        else:
            raise ValueError('self.status is not NotRun | Running | Done')

        for idx_seed in range(index_start, len(self.ls_seed)):
            try:
                model = self.model(self.ls_seed[idx_seed])
                
                self.ls_model.append(model)
                
                model.compile(**self.compile_kwargs)
                model.fit(*self.args, **self.kwargs)

                self.total_time += model.time_end - model.time_begin

            except KeyboardInterrupt as e:
                self.status = 'Running'
                self.set_attribute()

                save_result = saveModel(self, save_path)
                print('\n\nKeyboardInterrupt: ' +
                      save_result + ' model, model is not Done')
                traceback.print_exc()
                break
        else:
            self.set_attribute()
            self.status = 'Done'
            print('DONE!')
            print(saveModel(self, save_path))


def saveModel(model: MultiTimeModel, PATH: str, remove_tasks=True):
    '''
    `.mso`
    '''
    assert model.__class__.__name__ == 'MultiTimeModel'
    assert type(PATH) == str

    # check tail
    path_tmp = Path(PATH)
    index_dot = None
    for i in range(len(path_tmp.name) - 1, -1, -1):
        if path_tmp.name[i] == '.':
            index_dot = i
            break

    if index_dot is None:
        PATH += '.mso'
    else:
        assert path_tmp.name[i:] == '.mso', 'Only save model with .mso, not ' + \
            path_tmp.name[i:]

    model.__class__ = MultiTimeModel

    tasks = model.tasks 

    if remove_tasks is True:
        model.tasks = None
        model.compile_kwargs['tasks'] = None
        for submodel in model.ls_model:
            submodel.tasks = None
            submodel.last_pop.ls_tasks = None
            for subpop in submodel.last_pop:
                subpop.task = None
            if 'attr_tasks' in submodel.kwargs.keys():
                for attribute in submodel.kwargs['attr_tasks']:
                    # setattr(submodel, getattr(subm, name), None)
                    setattr(getattr(submodel, attribute), 'tasks', None)
                    pass
            else:
                submodel.crossover.tasks = None
                submodel.mutation.tasks = None

    try:
        f = open(PATH, 'wb')
        pickle.dump(model, f)
        f.close()

    except:
        cls = model.__class__
        model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

        if remove_tasks is True:
            model.tasks = tasks 
            model.compile_kwargs['tasks'] = None
            for submodel in model.ls_model:
                submodel.tasks = tasks
                submodel.last_pop.ls_tasks = tasks 
                for idx, subpop in enumerate(submodel.last_pop):
                    subpop.task = tasks[idx]
                if 'attr_tasks' in submodel.kwargs.keys():
                    for attribute in submodel.kwargs['attr_tasks']:
                        # setattr(submodel, getattr(subm, name), None)
                        setattr(getattr(submodel, attribute), 'tasks', tasks)
                        pass
                else:
                    submodel.crossover.tasks = tasks
                    submodel.mutation.tasks = tasks 
        return 'Cannot Saved'

    
    if remove_tasks is True:
        model.tasks = tasks 
        model.compile_kwargs['tasks'] = None
        for submodel in model.ls_model:
            submodel.tasks = tasks
            submodel.last_pop.ls_tasks = tasks 
            for idx, subpop in enumerate(submodel.last_pop):
                subpop.task = tasks[idx]
            if 'attr_tasks' in submodel.kwargs.keys():
                for attribute in submodel.kwargs['attr_tasks']:
                    # setattr(submodel, getattr(subm, name), None)
                    setattr(getattr(submodel, attribute), 'tasks', tasks)
                    pass
            else:
                submodel.crossover.tasks = tasks
                submodel.mutation.tasks = tasks 


    cls = model.__class__
    model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

    return 'Saved'


def loadModel(PATH: str, ls_tasks=None, set_attribute=False) -> AbstractModel:
    '''
    `.mso`
    '''
    assert type(PATH) == str

    # check tail
    path_tmp = Path(PATH)
    index_dot = None
    for i in range(len(path_tmp.name)):
        if path_tmp.name[i] == '.':
            index_dot = i
            break

    if index_dot is None:
        PATH += '.mso'
    else:
        assert path_tmp.name[i:] == '.mso', 'Only load model with .mso, not ' + \
            path_tmp.name[i:]

    f = open(PATH, 'rb')
    model = pickle.load(f)
    f.close()

    cls = model.__class__
    model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

    if model.tasks is None:
        model.tasks = ls_tasks
        if set_attribute is True:
            assert ls_tasks is not None, 'Pass ls_tasks plz!'
            model.compile_kwargs['tasks'] = ls_tasks
            for submodel in model.ls_model:
                submodel.tasks = ls_tasks
                submodel.last_pop.ls_tasks = ls_tasks
                for idx, subpop in enumerate(submodel.last_pop):
                    subpop.task = ls_tasks[idx]
                if 'attr_tasks' in submodel.kwargs.keys():
                    for attribute in submodel.kwargs['attr_tasks']:
                        # setattr(submodel, getattr(subm, name), None)
                        setattr(getattr(submodel, attribute),
                                'tasks', ls_tasks)
                        pass
                else:
                    submodel.crossover.tasks = ls_tasks
                    submodel.mutation.tasks = ls_tasks

                # submodel.search.tasks = ls_tasks
                # submodel.crossover.tasks = ls_tasks
                # submodel.mutation.tasks = ls_tasks

    return model


class CompareModel():
    # TODO so sánh
    def __init__(self, models: list[AbstractModel.model], label: list[str] = None) -> None:
        self.models = models
        if label is None:
            label = [m.name for m in self.models]
        else:
            assert len(self.models) == len(label)
            for idx in range(len(label)):
                if label[idx] == Ellipsis:
                    label[idx] = self.models[idx].name

        self.label = label

    def render(self, shape: tuple = None, min_cost=0, nb_generations: int = None, step=1, figsize: tuple[int, int] = None, dpi=200, yscale: str = None, re=False, label_shape=None, label_loc=None):
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


class TuningModel:
    def __init__(self, model_name, nb_run: int = 1, list_parameter: list[tuple] = []) -> None:
        self.best_compile_parameter = {}
        self.best_fit_parameter = {}
        self.model_name = model_name
        self.list_parameter: list[tuple(str, list)] = list_parameter
        self.nb_run = nb_run

        self.default_lsmodel= None 

    def compile(self, ls_benchmark=None, benchmark_weights=[], name_benchmark = [], ls_IndClass = [],  **kwargs):
        # if ls_benchmark is None:
        #     ls_benchmark.append(kwargs['tasks'])
        #     ls_IndClass.append(kwargs['IndClass'])
        #     name_benchmark.append("default")
        # else:
        #     if kwargs['tasks'] not in ls_benchmark and kwargs['tasks'] is not None:
        #         ls_benchmark.append(kwargs['tasks']) 
        #         ls_IndClass.append(kwargs['IndClass'])
        #         name_benchmark.append("default")

        assert len(ls_benchmark) == len(
            benchmark_weights), 'len of ls benchmark and benchmark_weights must be same'
        assert np.sum(np.array(benchmark_weights)
                      ) == 1, 'Benchmark weighs need sum up to 1'

        self.compile_kwargs = kwargs
        self.ls_benchmark: list[list[AbstractTask]] = ls_benchmark
        self.benchmark_weights = benchmark_weights
        self.name_benchmark = name_benchmark
        self.ls_IndClass = ls_IndClass

    def fit_multibenchmark(self, curr_fit_parameter, curr_compile_parameter, nb_run=1, save_path="./RESULTS/tuning_smp/k", name_model="model.mso"):
        ls_model = []
        for idx, benchmark in enumerate(self.ls_benchmark):
            model = MultiTimeModel(self.model_name)
            curr_compile_parameter['tasks'] = benchmark
            curr_compile_parameter['IndClass'] = self.ls_IndClass[idx] 
            model.compile(
                **curr_compile_parameter
            )
            model.fit(
                **curr_fit_parameter
            )

            if os.path.isdir(save_path) is False:
                os.makedirs(save_path)

            model.run(
                nb_run=self.nb_run,
                save_path=save_path + self.name_benchmark[idx]
            )
            model = loadModel(save_path + self.name_benchmark[idx] , ls_tasks=benchmark, set_attribute= True)

            import json
            file = open(save_path + self.name_benchmark[idx] + "_" + name_model.split('.')[0] + "_result.txt", 'w')
            file.write(json.dumps(dict(enumerate(model.history_cost[-1]))))
            file.close()   
            ls_model.append(model)
        
        if self.default_lsmodel is None: 
            self.default_lsmodel = ls_model 

        return ls_model 



    def fit(self, curr_fit_parameter, curr_compile_parameter, nb_run=1, save_path="./RESULTS/tuning_smp/", name_model="model.mso"):
        model = MultiTimeModel(self.model_name)
        model.compile(
            **curr_compile_parameter
        )
        model.fit(
            **curr_fit_parameter
        )
        if os.path.isdir(save_path) is False:
            os.makedirs(save_path)
        ls_tasks = model.tasks
        model.run(
            nb_run=self.nb_run,
            save_path=save_path + name_model
        )

        model = loadModel(save_path + name_model,
                          ls_tasks=ls_tasks, set_attribute=True)

        import json
        file = open(save_path + name_model.split('.')[0] + "_result.txt", 'w')
        file.write(json.dumps(dict(enumerate(model.history_cost[-1]))))
        file.close()

        return model
    
    def compare_between_2_ls_model(self, ls_model1: list[AbstractModel.model], ls_model2 : list[AbstractModel.model], min_value= 0, take_point = False):
        '''
        compare the result between models and return best model 
        [[model1_cec, model1_gecco], [model2_cec, model2_gecco]]
        '''
        point_model = np.zeros(shape= (2,))
        for benchmark in range(len(ls_model1)):
            result_model1 = np.where(ls_model1[benchmark].history_cost[-1] > min_value, ls_model1[benchmark].history_cost[-1], min_value)
            result_model2 = np.where(ls_model2[benchmark].history_cost[-1] > min_value, ls_model2[benchmark].history_cost[-1], min_value) 

            point1 =  np.sum(result_model1 < result_model2) / len(ls_model1[benchmark].tasks)
            point2 = np.sum(result_model2 < result_model1) / len(ls_model1[benchmark].tasks)

            point_model[0] += point1 * self.benchmark_weights[benchmark]
            point_model[1] += point2 * self.benchmark_weights[benchmark]
            point_model += (1 - np.sum(point_model))/2 
        if take_point is True: 
            return point_model
        else: 
            return np.argmax(point_model) 


    def take_idx_best_lsmodel(self, set_ls_model: list[list[AbstractModel.model]], min_value = 0, compare_default = True, take_point = False):
        if compare_default is True: 
            ls_point = [] 
            for idx, ls_model in enumerate(set_ls_model):
                ls_point += [(self.compare_between_2_ls_model(ls_model, self.default_lsmodel, min_value,take_point= True)[0])]
            if take_point is True:
                return ls_point 
            else: 
                best_idx = np.argmax(np.array(ls_point))
                return best_idx
            pass 
        else: 
            best_idx = 0  
            for idx, ls_model in enumerate(set_ls_model[1:],start= 1 ):
                better_idx = self.compare_between_2_ls_model(set_ls_model[best_idx], ls_model, min_value)
                if better_idx == 1: 
                    best_idx = idx 
        
        return best_idx 

    def take_idx_best_model(self, ls_model) -> int:
        compare = CompareModel(ls_model)
        end = compare.detail_compare_result()
        return np.argmax([float(point.split("/")[0]) for point in end.iloc[-1]])

    def run(self, path="./RESULTS/tuning", replace_folder=False,min_value = 0,  **kwargs):
        if path[-1] != "/":
            path += "/"
        path = path + self.model_name.__name__.split('.')[-1]
        curr_fit_parameter = kwargs.copy()
        curr_compile_parameter = self.compile_kwargs.copy()

        self.best_compile_parameter = self.compile_kwargs.copy()
        self.best_fit_parameter = kwargs.copy()
        result = self.list_parameter.copy()
        result = [list(result[i]) for i in range(len(result))]

        # folder
        if os.path.isdir(path) is True:
            if replace_folder is True:
                pass
            else:
                raise Exception("Folder is existed")
        else:
            os.makedirs(path)

        self.root_folder = os.path.join(path)

        # check value pass
        for name_arg, arg_pass in self.list_parameter:
            # name_args: 'crossover' , arg_pass: {'gamma': []}
            if name_arg in curr_compile_parameter.keys():
                if callable(curr_compile_parameter[name_arg]):
                    for name_para, para_value in arg_pass.items():
                        # name_para: 'gamma'
                        # para_value: [0.4, 0.5]
                        curr_para_value = getattr(
                            curr_compile_parameter[name_arg], name_para)
                        if curr_para_value is not None:
                            if curr_para_value in para_value:
                                para_value.remove(curr_para_value)
                                para_value.insert(0, curr_para_value)
                            else:
                                para_value.insert(0, curr_para_value)
                        else:
                            pass
                            setattr(
                                curr_compile_parameter[name_arg], name_para, para_value[0])
                else:  # if arg_pass in self.compile_kwargs is a str/number
                    if curr_compile_parameter[name_arg] is not None:
                        if curr_compile_parameter[name_arg] in arg_pass:
                            arg_pass.remove(curr_compile_parameter[name_arg])
                            arg_pass.insert(
                                0, curr_compile_parameter[name_arg])
                        else:
                            arg_pass.insert(
                                0, curr_compile_parameter[name_arg])
                    else:
                        curr_compile_parameter[name_arg] = arg_pass[0]

            elif name_arg in curr_fit_parameter.keys():
                if callable(curr_fit_parameter[name_arg]):
                    for name_para, para_value in arg_pass.items():
                        curr_para_value = getattr(
                            curr_fit_parameter[name_arg], name_para)
                        if curr_para_value is not None:
                            if curr_para_value in para_value:
                                para_value.remove(curr_para_value)
                                para_value.insert(0, curr_para_value)
                            else:
                                para_value.insert(0, curr_para_value)
                        else:
                            pass
                            setattr(
                                curr_fit_parameter[name_arg], name_para, para_value[0])

                else:
                    if curr_fit_parameter[name_arg] is not None:
                        if curr_fit_parameter[name_arg] in arg_pass:
                            arg_pass.remove(curr_fit_parameter[name_arg])
                            arg_pass.insert(0, curr_fit_parameter[name_arg])
                        else:
                            arg_pass.insert(0, curr_fit_parameter[name_arg])
                    else:
                        pass
                        curr_fit_parameter[name_arg] = arg_pass[0]

        
        # run many 
        curr_order_params = 0
        for name_arg, arg_value in self.list_parameter:
            curr_order_params += 1
            idx = self.list_parameter.index((name_arg, arg_value))

            if name_arg in self.compile_kwargs.keys():
                print("\n",name_arg)
                if callable(self.compile_kwargs[name_arg]):
                    for name_para, para_value in arg_value.items():
                        print("\n",name_para)
                        # take each parameter in function
                        # name_para: 'gamma'
                        # para_value: [0.4, 0.6]
                        # TODO
                        sub_folder = self.get_curr_path_folder(
                            curr_order_params) + "/" + str(name_para)
                        # root_folder/gamma

                        assert type(para_value) == list
                        curr_compile_parameter = self.best_compile_parameter.copy()
                        set_ls_model = []
                        for value in para_value:
                            value_folder_path = sub_folder + \
                                "/" + str(value) + "/"
                            # root_folder/gamma/0.4/
                            print(value)
                            setattr(curr_compile_parameter[name_arg], name_para, value)
                            # ls_model.append(self.fit(
                            #     self.best_fit_parameter, curr_compile_parameter, save_path=value_folder_path))
                            set_ls_model.append(self.fit_multibenchmark(self.best_fit_parameter, curr_compile_parameter, save_path=value_folder_path))
                            # set_ls_model.append(self.fit_multibenchmark(self.best_fit_parameter, curr_compile_parameter))

                        # TODO: take the best model and update best parameter
                        ls_point = self.take_idx_best_lsmodel(set_ls_model, min_value= min_value, take_point=True)
                        value = para_value[np.argmax(ls_point)]


                        import json 
                        # file= open(value_folder_path + "/result.txt" , 'w')
                        file = open(sub_folder + "/" + "result.txt", 'w')
                        file.write(name_arg+" ") 
                        file.write(name_para)
                        file.write(json.dumps(list(zip(arg_value[name_para], ls_point))))
                        file.close() 
                        
                        
                        
                        setattr(
                            self.best_compile_parameter[name_arg], name_para, value)

                        # save result

                        result[idx][1][name_para] = value

                else:
                    # if self.complile_kwargs[name_arg] is str/ number
                    assert type(arg_value) == list
                    curr_compile_parameter = self.best_compile_parameter.copy()
                    set_ls_model = []
                    sub_folder = self.get_curr_path_folder(
                        curr_order_params) + "/" + str(name_arg)
                    # root_folder/lr
                    for value in arg_value:
                        value_folder_path = sub_folder + "/" + str(value) + "/"
                        # root_folder/lr/0.1/
                        print(value)
                        curr_compile_parameter[name_arg] = value
                        # self.fit(self.best_fit_parameter, curr_compile_parameter)
                        set_ls_model.append(self.fit_multibenchmark(
                            self.best_fit_parameter, curr_compile_parameter, save_path=value_folder_path))
                    # TODO: take the best model and update best parameter
                    ls_point = self.take_idx_best_lsmodel(set_ls_model, min_value= min_value, take_point=True)
                    value = arg_value[np.argmax(ls_point)]

                    self.best_compile_parameter[name_arg] = value

                    import json 
                    # file = open(value_folder_path + "/result.txt", 'w')
                    file = open(sub_folder + "/" + "result.txt", 'w')
                    file.write(name_arg)
                    file.write(json.dumps(list(zip(arg_value, ls_point))))
                    file.close() 

                    # save result
                    result[idx][1] = value

            elif name_arg in curr_fit_parameter.keys():
                print("\n",name_arg)
                if callable(curr_fit_parameter[name_arg]):
                    for name_para, para_value in arg_value.items():
                        print("\n",name_para)

                        sub_folder = self.get_curr_path_folder(
                            curr_order_params) + "/" + str(name_para)

                        assert type(para_value) == list
                        set_ls_model = []
                        curr_fit_parameter = self.best_fit_parameter.copy()
                        for value in para_value:
                            value_folder_path = sub_folder + \
                                "/" + str(value) + "/"
                            print(value)
                            setattr(
                                curr_fit_parameter[name_arg], name_para, value)
                            set_ls_model.append(self.fit_multibenchmark(
                                curr_fit_parameter, self.best_compile_parameter, save_path=value_folder_path))
                        # TODO: take the best modle in update best parameter
                        ls_point = self.take_idx_best_lsmodel(set_ls_model, min_value= min_value, take_point=True)
                    
                        value = para_value[np.argmax(ls_point)]
                        setattr(
                            self.best_fit_parameter[name_arg], name_arg, value)

                        import json 
                        # file= open(value_folder_path + "/result.txt" , 'w')
                        file = open(sub_folder + "/" + "result.txt", 'w')
                        file.write(name_arg) 
                        file.write(name_para)
                        file.write(json.dumps(list(zip(arg_value[name_para], ls_point))))
                        file.close() 

                        # save result
                        result[idx][1][name_para] = value
                else:
                    assert type(arg_value) == list
                    curr_fit_parameter = self.best_fit_parameter.copy()
                    set_ls_model = []
                    sub_folder = self.get_curr_path_folder(
                        curr_order_params) + "/" + str(name_arg)
                    for value in arg_value:
                        print(value)
                        value_folder_path = sub_folder + "/" + str(value) + "/"
                        curr_fit_parameter[name_arg] = value
                        # self.fit(curr_fit_parameter, self.best_compile_parameter)
                        set_ls_model.append(self.fit_multibenchmark(
                            curr_fit_parameter, self.best_compile_parameter, save_path=value_folder_path))
                    # TODO: take the best model and update best fit parameter
                    ls_point = self.take_idx_best_lsmodel(set_ls_model, min_value= min_value, take_point=True)
                    value = arg_value[np.argmax(ls_point)]

                    self.best_fit_parameter[name_arg] = value

                    import json 
                    file = open(sub_folder + "/" + "result.txt", 'w')
                    file.write(name_arg)
                    file.write(json.dumps(list(zip(arg_value, ls_point))))
                    file.close() 


                    # save result
                    result[idx][1] = value

        import json
        file = open(self.root_folder + "/result.txt", 'w')
        file.write(json.dumps(result))
        file.close()

        return self.best_fit_parameter, self.best_compile_parameter, result

    def get_curr_path_folder(self, curr_order_params):
        if curr_order_params == 1:
            # path = root_path/gamma/
            path = self.root_folder
            pass
        else:
            # path = root_path/gamma/"value_gamma"
            path = self.root_folder
            index = 1
            while(curr_order_params > index):
                name_arg, arg_value = self.list_parameter[index-1]
                if type(arg_value) != list:
                    for key, _ in arg_value.items():
                        index += 1
                        # find value
                        if name_arg in self.best_compile_parameter.keys():
                            value = getattr(
                                self.best_compile_parameter[name_arg], key)
                        else:
                            value = getattr(
                                self.best_fit_parameter[name_arg], key)

                        # update path
                        path += "/" + str(key) + "/" + str(value)
                else:
                    if name_arg in self.best_compile_parameter.keys():
                        value = self.best_compile_parameter[name_arg]
                    else:
                        value = self.best_fit_parameter[name_arg]

                    # update path
                    path += "/" + str(name_arg) + "/" + str(value)
                    index += 1
        return path

class MultiBenchmark(): 
    def __init__(self, ls_benchmark = [], name_benchmark = [], ls_IndClass = [], model : AbstractModel= None) :
        self.ls_benchmark = ls_benchmark 
        self.ls_name_benchmark = name_benchmark 
        self.ls_IndClass = ls_IndClass 

        self.model = model  
        pass
    
    def compile(self, **kwargs): 
        self.compile_kwargs = kwargs 
    
    def fit(self, **kwargs):
        self.fit_kwargs = kwargs 
    
    def run(self,nb_run = 1, save_path = './RESULTS/result/'): 
        self.ls_model = [] 
        for idx, benchmark in enumerate(self.ls_benchmark): 
            self.compile_kwargs['tasks'] = benchmark 
            self.compile_kwargs['IndClass'] = self.ls_IndClass[idx] 

            model = MultiTimeModel(model = self.model) 
            model.compile(**self.compile_kwargs) 
            model.fit(**self.fit_kwargs) 
            model.run(nb_run = nb_run, save_path= save_path + str(self.ls_name_benchmark[idx]) + ".mso") 
            self.ls_model.append(model) 
    
    def print_result(self,print_attr = [],print_name = True, print_time = False, print_name_attr = False):
        
        for idx, model in enumerate(self.ls_model): 
            if print_name : 
                print(self.ls_name_benchmark[idx])
            model.print_result(print_attr, print_time= print_time, print_name= print_name_attr) 

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
                

            
            
    