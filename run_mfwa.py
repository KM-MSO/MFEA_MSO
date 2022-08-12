from MFEA_lib.model import mfwa
from MFEA_lib.model.utils import *
from MFEA_lib.operators.Crossover import *
from MFEA_lib.operators.Mutation import *
from MFEA_lib.operators.Selection import *
from MFEA_lib.tasks.Benchmark.Funcs import *

tasks, IndClass = WCCI22_benchmark.get_complex_benchmark(1)

baseModel = mfwa.model()
baseModel.compile(
    IndClass= IndClass,
    tasks= tasks,
    crossover= SBX_Crossover(nc = 2),
    mutation= Polynomial_Mutation(nm = 5),
    selection= ElitismSelection()
)
solve = baseModel.fit(
    nb_generations=1000,
    diameter=1, 
    S_param=100
)