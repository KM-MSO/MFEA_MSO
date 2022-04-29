import numpy as np
from MFEA_lib.operators.Crossover import PMX_Crossover
from MFEA_lib.tasks.Benchmark.IDPC_EDU import IDPC_EDU_benchmark
a = ['a','d', 'b','c']
print(sorted(a))
# cost = []
# tasks = IDPC_EDU_benchmark.get_tasks(1)[0]
# for i in range(100):
    

#     idx =  np.random.randint(10)
#     task = tasks[idx]
#     sol = np.random.permutation(90).reshape(2,45)
#     cost.append(task(sol))
# print(cost, np.array(cost).mean(), np.array(cost).min())
