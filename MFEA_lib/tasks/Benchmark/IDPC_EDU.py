import os
import numpy as np
from MFEA_lib.tasks.task import create_idpc, IDPC_EDU_FUNC
from ...EA import Individual
from tqdm import tqdm
import ray
path = os.path.dirname(os.path.realpath(__file__))

class Ind_EDU(Individual):
    def __init__(self, genes, dim=None):
        super().__init__(genes, dim)
        if genes is None:
            self.genes: np.ndarray = np.append([np.random.permutation(dim)], [np.random.randint(0, dim, dim)], axis= 0)


class IDPC_EDU_benchmark:
    def get_tasks(ID_set: int):
        print('\rReading data...')
        file_list = sorted(os.listdir(path + '/__references__/IDPC_DU/IDPC_EDU/data/set' + str(ID_set)))
        tasks = [create_idpc(path + '/__references__/IDPC_DU/IDPC_EDU/data/set' + str(ID_set), f) for f in file_list]
        #tasks = ray.get([create_idpc.remote(path + '/__references__/IDPC_DU/IDPC_EDU/data/set' + str(ID_set), f) for f in file_list] )
        print(f'\rNumber of task:{len(tasks)}')
        return sorted(tasks, key = lambda t: t.file), Ind_EDU

