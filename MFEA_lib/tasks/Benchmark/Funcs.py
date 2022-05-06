import numpy as np
from typing import Tuple, Type
from ...EA import Individual
from ..function import AbstractFunc, Sphere, Weierstrass, Ackley, Rosenbrock, Schwefel, Griewank, Rastrigin
from scipy.io import loadmat
import os
from ..cec14_func import *
path = os.path.dirname(os.path.realpath(__file__))

class Individual_func(Individual):
    def __init__(self, genes, dim= None) -> None:
        super().__init__(genes, dim)
        if genes is None:
            self.genes: np.ndarray = np.random.rand(dim)

class GECCO20_benchmark_50tasks():
    task_size = 50
    dim = 50

    def get_choice_function(ID) -> list[int]:
        choice_functions = []
        if ID == 1:
            choice_functions = [1]
        elif ID == 2:
            choice_functions = [2]
        elif ID == 3:
            choice_functions = [4]
        elif ID == 4:
            choice_functions = [1, 2, 3]
        elif ID == 5:
            choice_functions = [4, 5, 6]
        elif ID == 6:
            choice_functions = [2, 5, 7]
        elif ID == 7:
            choice_functions = [3, 4, 6]
        elif ID == 8:
            choice_functions = [2, 3, 4, 5, 6]
        elif ID == 9:
            choice_functions = [2, 3, 4, 5, 6, 7]
        elif ID == 10:
            choice_functions = [3, 4, 5, 6, 7]
        else:
            raise ValueError("Invalid input: ID should be in [1,10]")
        return choice_functions

    def get_items(ID, fix = False) -> Tuple[list[AbstractFunc], Type[Individual_func]]:
        choice_functions = __class__.get_choice_function(ID)

        tasks = []

        for task_id in range(__class__.task_size):
            func_id = choice_functions[task_id % len(choice_functions)]
            file_dir = path + "/__references__/GECCO20/Tasks/benchmark_" + str(ID)
            shift_file = "/bias_" + str(task_id + 1)
            rotation_file = "/matrix_" + str(task_id + 1)
            matrix = np.loadtxt(file_dir + rotation_file)
            shift = np.loadtxt(file_dir + shift_file)

            if func_id == 1:
                tasks.append(
                    Sphere(__class__.dim, shift= shift, rotation_matrix= matrix,bound= [-100, 100])
                )
            elif func_id == 2:
                tasks.append(
                    Rosenbrock(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-50, 50])
                )
            elif func_id == 3:
                tasks.append(
                    Ackley(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-50, 50])
                )
            elif func_id == 4:
                tasks.append(
                    Rastrigin(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-50, 50])
                )
            elif func_id == 5:
                tasks.append(
                    Griewank(__class__.dim, shift= shift, rotation_matrix= matrix, bound = [-100, 100])
                )
            elif func_id == 6:
                tasks.append(
                    Weierstrass(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-0.5, 0.5])
                )
            elif func_id == 7:
                tasks.append(
                    Schwefel(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-500, 500], fixed= fix)
                )
        return tasks, Individual_func

class CEC17_benchmark():
    dim = 50
    task_size = 2

    def get_10tasks_benchmark(fix = False)-> Tuple[list[AbstractFunc], Type[Individual_func]]:
        tasks = [
        Sphere(     50,shift= 0,    bound= [-100, 100]),   # 0
        Sphere(     50,shift= 80,   bound= [-100, 100]),  # 80
        Sphere(     50,shift= -80,  bound= [-100, 100]), # -80
        Weierstrass(25,shift= -0.4, bound= [-0.5, 0.5]), # -0.4
        Rosenbrock( 50,shift= -1,   bound= [-50, 50]),# 0
        Ackley(     50,shift= 40,   bound= [-50, 50]),    # 40
        Weierstrass(50,shift= -0.4, bound= [-0.5, 0.5]), # -0.4
        Schwefel(   50,shift= 0,    bound= [-500, 500], fixed = fix), # 420.9687
        Griewank(   50,shift= [-80, 80], bound= [-100, 100]), # -80, 80
        Rastrigin(  50,shift= [40, -40], bound= [-50, 50]),# -40, 40
        ]
        return tasks, Individual_func


    def get_2tasks_benchmark(ID)-> Tuple[list[AbstractFunc], Type[Individual_func]]:
        #TODO
        tasks = []

        if ID == 1:
            ci_h = loadmat(path + "/__references__/CEC17/Tasks/CI_H.mat")
            tasks.append(
                Griewank(
                    dim= 50,
                    shift = ci_h['GO_Task1'],
                    rotation_matrix= ci_h['Rotation_Task1'],
                    bound= (-100, 100)
                )
            )
            tasks.append(
                Rastrigin(
                    dim= 50,
                    shift= ci_h['GO_Task2'],
                    rotation_matrix= ci_h['Rotation_Task2'],
                    bound= (-50, 50)
                )
            )
        elif ID == 2:
            ci_m = loadmat(path + "/__references__/CEC17/Tasks/CI_M.mat")
            tasks.append(
                Ackley(
                    dim= 50,
                    shift= ci_m['GO_Task1'],
                    rotation_matrix= ci_m['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Rastrigin(
                    dim= 50,
                    shift= ci_m['GO_Task2'],
                    rotation_matrix= ci_m['Rotation_Task2'],
                    bound= (-50, 50)
                )
            )
        elif ID == 3:
            ci_l = loadmat(path + "/__references__/CEC17/Tasks/CI_L.mat")
            tasks.append(
                Ackley(
                    dim= 50,
                    shift= ci_l['GO_Task1'],
                    rotation_matrix= ci_l['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Schwefel(
                    dim= 50,
                    bound= (-500, 500)
                )
            )
        elif ID == 4:
            pi_h = loadmat(path + "/__references__/CEC17/Tasks/PI_H.mat")
            tasks.append(
                Rastrigin(
                    dim= 50, 
                    shift= pi_h['GO_Task1'],
                    rotation_matrix= pi_h['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Sphere(
                    dim= 50, 
                    shift = pi_h['GO_Task2'],
                    bound= (-100, 100)
                )
            )
        elif ID == 5:
            pi_m = loadmat(path + "/__references__/CEC17/Tasks/PI_M.mat")
            tasks.append(
                Ackley(
                    dim= 50, 
                    shift= pi_m['GO_Task1'],
                    rotation_matrix= pi_m['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Rosenbrock(
                    dim= 50, 
                    bound= (-50, 50)
                )
            )
        elif ID == 6:
            pi_l = loadmat(path + "/__references__/CEC17/Tasks/PI_L.mat")
            tasks.append(
                Ackley(
                    dim= 50, 
                    shift= pi_l['GO_Task1'],
                    rotation_matrix= pi_l['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Weierstrass(
                    dim= 25, 
                    shift = pi_l['GO_Task2'],
                    rotation_matrix= pi_l['Rotation_Task2'],
                    bound= (-0.5, 0.5)
                )
            )
        elif ID == 7:
            ni_h = loadmat(path + "/__references__/CEC17/Tasks/NI_H.mat")
            tasks.append(
                Rosenbrock(
                    dim= 50, 
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Rastrigin(
                    dim= 50, 
                    shift = ni_h['GO_Task2'],
                    rotation_matrix= ni_h['Rotation_Task2'],
                    bound= (-50, 50)
                )
            )
        elif ID == 8:
            ni_m = loadmat(path + "/__references__/CEC17/Tasks/NI_M.mat")
            tasks.append(
                Griewank(
                    dim= 50, 
                    shift= ni_m['GO_Task1'],
                    rotation_matrix= ni_m['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Weierstrass(
                    dim= 50, 
                    shift = ni_m['GO_Task2'],
                    rotation_matrix= ni_m['Rotation_Task2'],
                    bound= (-0.5, 0.5)
                )
            )
        elif ID == 9:
            ni_l = loadmat(path + "/__references__/CEC17/Tasks/NI_L.mat")
            tasks.append(
                Rastrigin(
                    dim= 50, 
                    shift= ni_l['GO_Task1'],
                    rotation_matrix= ni_l['Rotation_Task1'],
                    bound= (-50, 50)
                )
            )
            tasks.append(
                Schwefel(
                    dim= 50, 
                    bound= (-500, 500)
                )
            )
        else:
            raise ValueError('ID must be an integer from 1 to 9, not ' + ID)
        return tasks, Individual_func

class WCCI22_benchmark():
    def get_50tasks_benchmark(ID) -> list[AbstractFunc]:
        return GECCO20_benchmark_50tasks.get_items(ID)
    
    class Task():
        def __init__(self,dim,func_id,Ub,Lb, shift: list = 0, rotation_matrix: np.ndarray = None,SS:list =[]):
            self.dim = dim
            self.shift =shift
            self.rotation_matrix =rotation_matrix
            self.Ub =Ub
            self.Lb = Lb
            self.SS = SS
            self.func_id =func_id
            self.name = "func_id = " + str(func_id)
        def decode(self,x,sh_rate,rotation_matrix,shift):
            x_decode = x-shift
            x_decode = x_decode*sh_rate
            x_decode= rotation_matrix @ x_decode
            if len(self.SS) != 0:
                y= np.zeros(self.dim) 
                for i in range(self.dim):
                    y[i] = x_decode[self.SS[i]-1]
                return y
            return x_decode 
        @jit(nopython = True)
        def func(x,ID,dim):
            if ID == 1: 
                return Ellips_func(x,dim) + 100
            if ID == 2: 
                return Ben_cigar_func(x,dim) + 200
            if ID == 3: 
                return Discus_func(x,dim) + 300
            if ID == 4:
                return Rosenbrock_func(x,dim) + 400
            if ID == 5: 
                return Ackley_func(x,dim) + 500
            if ID == 6:
                return Weierstrass_func(x,dim) + 600
            if ID == 7:
                return Griewank_func(x,dim) + 700
            if ID == 8: 
                return Rastrigin_func(x,dim) + 800
            if ID == 9:
                return Rastrigin_func(x,dim) +900
            if ID == 10:
                return Schwefel_func(x,dim) + 1000
            if ID == 11:
                return Schwefel_func(x,dim) + 1100
            if ID == 12:
                return Katsuura_func(x,dim) + 1200
            if ID == 13: 
                return HappyCat_func(x,dim) + 1300
            if ID == 14:
                return Hgbat_func(x,dim) + 1400
            if ID == 15:
                return Grie_rosen_func(x,dim) + 1500
            if ID == 16:
                return Escaffer6_func(x,dim) + 1600
            if ID == 17:
                return hf01(x,dim) + 1700
            if ID == 18:
                return hf02(x,dim) + 1800
            if ID == 19:
                return hf03(x,dim) + 1900
            if ID == 20:
                return hf04(x,dim) + 2000
            if ID == 21: 
                return hf05(x,dim) + 2100
            if ID == 22:
                return hf06(x,dim) + 2200 

        def fnceval(self,x):
            if self.func_id < 4 or self.func_id == 5 or self.func_id >15:
                x = self.decode(x,1,self.rotation_matrix,self.shift)
            if self.func_id == 4:
                x= self.decode(x, 2.048 / 100.0,self.rotation_matrix,self.shift)
            if self.func_id == 6:
                x= self.decode(x,0.5/100,self.rotation_matrix,self.shift)
            if self.func_id == 7: 
                x=self.decode(x,600/100,self.rotation_matrix,self.shift)
            if self.func_id == 8:
                x= self.decode(x,5.12/100,np.identity,self.shift)
            if self.func_id == 9:
                x =self.decode(x,5.12/100,self.rotation_matrix,self.shift)
            if self.func_id == 10:
                x =self.decode(x,10,np.identity,self.shift)
            if self.func_id == 11:
                x = self.decode(x,10,self.rotation_matrix,self.shift)
            if self.func_id > 11 and self.func_id <16:
                x = self.decode(x,5/100,self.rotation_matrix,self.shift)
            return __class__.func(x,self.func_id,self.dim)
        def __call__(self,x):
            x= x*(self.Ub-self.Lb)+self.Lb
            return self.fnceval(x)

    def get_complex_benchmark(ID) -> list[AbstractFunc]:
        dim =50  
        task =[]
        file_dir = path + "/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/benchmark_" + str(ID)
        for i in range(1,3):
            shift_file = "/bias_" + str(i)
            rotation_file = "/matrix_" + str(i)
            matrix = np.loadtxt(file_dir + rotation_file)
            shift = np.loadtxt(file_dir + shift_file)
            if ID == 1:
                task.append(__class__.Task(50,6,-100,100,shift,matrix))
            if ID == 2:
                task.append(__class__.Task(50,7,-100,100,shift,matrix))
            if ID == 3: 
                task.append(__class__.Task(50,17,-100,100,shift,matrix))
                f = path+'/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/shuffle/shuffle_data_' + str(17) + "_D" + str(dim)+'.txt'
                task[i-1].SS = np.loadtxt(f).astype(int)
            if ID == 4:
                task.append(__class__.Task(50,13,-100,100,shift,matrix))
            if ID == 5: 
                task.append(__class__.Task(50,15,-100,100,shift,matrix))
            if ID == 6:
                task.append(__class__.Task(50,21,-100,100,shift,matrix))
                f = path+'/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/shuffle/shuffle_data_' + str(21) + "_D" + str(dim)+'.txt'
                task[i-1].SS = np.loadtxt(f).astype(int)
            if ID == 7:
                task.append(__class__.Task(50,22,-100,100,shift,matrix))
                f = path+'/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/shuffle/shuffle_data_' + str(22) + "_D" + str(dim)+'.txt'
                task[i-1].SS = np.loadtxt(f).astype(int)
            if ID == 8:
                task.append(__class__.Task(50,5,-100,100,shift,matrix))
            if ID == 9:
                if i == 1:
                    task.append(__class__.Task(50,11,-100,100,shift,matrix))
                else :
                    task.append(__class__.Task(50,16,-100,100,shift,matrix))
            if ID == 10:
                if i == 1:
                    task.append(__class__.Task(50,20,-100,100,shift,matrix))
                    f = path+'/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/shuffle/shuffle_data_' + str(20) + "_D" + str(dim)+'.txt'
                    task[0].SS = np.loadtxt(f).astype(int)
                else:
                    task.append(__class__.Task(50,21,-100,100,shift,matrix))
                    f = path+'/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/shuffle/shuffle_data_' + str(21) + "_D" + str(dim)+'.txt'
                    task[1].SS = np.loadtxt(f).astype(int)
        return task, Individual_func