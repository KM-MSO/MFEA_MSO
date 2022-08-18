import os
import numpy as np
from tqdm import tqdm
from MFEA_lib.tasks.task import IDPC_EDU_FUNC,IDPC_EDU,IDPC_EDU_new
from ...EA import Individual
path = os.path.dirname(os.path.realpath(__file__))

class Ind_EDU(Individual):
    def __init__(self, genes, dim=None) -> None:
        super().__init__(genes, dim)
        if genes is None:
            self.genes: np.ndarray = np.append([np.random.permutation(dim)], [np.random.randint(0, dim, dim)], axis= 0)


class IDPC_EDU_benchmark:
    def get_tasks(ID_set: int):
        print('\rReading data...')
        tasks = []
        file_list = sorted(os.listdir(path + '/__references__/IDPC_DU/IDPC_EDU/data/set' + str(ID_set)))
        for file_name in tqdm(file_list):
            tasks.append(IDPC_EDU_FUNC(path + '/__references__/IDPC_DU/IDPC_EDU/data/set' + str(ID_set), file_name))
        return tasks, Ind_EDU

class IDPC_EDU_benchmark_1:
    def get_tasks(ID_set: int):
        print('\rReading data...')
        tasks = []
        file_list = sorted(os.listdir(path + '/__references__/IDPC_DU/IDPC_EDU/data/set' + str(ID_set)))[:10]
        for file_name in tqdm(file_list):
            tasks.append(IDPC_EDU(path + '/__references__/IDPC_DU/IDPC_EDU/data/set' + str(ID_set), file_name))
        return tasks
class IDPC_EDU_benchmark_2:
    def get_tasks(ID_set: int):
        print('\rReading data...')
        tasks = []
        file_list = sorted(os.listdir(path + '/__references__/IDPC_DU/IDPC_EDU/data/set' + str(ID_set)))[:10]
        for file_name in tqdm(file_list):
            tasks.append(IDPC_EDU_new(path + '/__references__/IDPC_DU/IDPC_EDU/data/set' + str(ID_set), file_name))
        return tasks