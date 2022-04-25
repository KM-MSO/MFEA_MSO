import numpy as np
from MFEA_lib.operators.Crossover import PMX_Crossover
from MFEA_lib.tasks.Benchmark.IDPC_EDU import IDPC_EDU_benchmark
p1 = np.array([9,8,6,4,2,3,7,1,5]) - 1
p2 = np.array([7,5,2,9,4,8,6,3,1]) - 1
func = PMX_Crossover()
tasks, IndClass = IDPC_EDU_benchmark.get_tasks(1)
func.getInforTasks(IndClass, tasks)
func.dim_uss = 9
print(func(p1,p2))
from tqdm import tqdm
for i in tqdm(range(500 * 100 * 500)):
    func(p1, p2)

