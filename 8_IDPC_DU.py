#!/usr/bin/env python
# coding: utf-8

# In[1]:


from MFEA_lib.tasks.Benchmark.IDPC_EDU import IDPC_EDU_benchmark
from MFEA_lib.model import MFEA_base, MFEA_surrogate
from MFEA_lib.model.utils import *
from MFEA_lib.operators.Crossover import *
from MFEA_lib.operators.Mutation import *
from MFEA_lib.operators.Selection import *
from MFEA_lib.tasks.surrogate import SurrogatePipeline
import time
import ray
ray.init()
s = time.time()
tasks, IndClass = IDPC_EDU_benchmark.get_tasks(1)
print(time.time() - s)


baseModel = MFEA_base.betterModel()
baseModel.compile(
    IndClass= IndClass,
    tasks= tasks,
    crossover= IDPCEDU_Crossover(),
    mutation= IDPCEDU_Mutation(),
    selection= ElitismSelection(),
    surrogate_pipeline = SurrogatePipeline(10, 10,10,10),
)
solve = baseModel.fit(
    nb_generations = 1000, rmp = 0.3, nb_inds_each_task= 100, 
    bound_pop= [0, 1], evaluate_initial_skillFactor= True
)

