from ortools.linear_solver import pywraplp
import random
import matplotlib.pyplot as plt
import numpy as np


  
n = 50
#early thuoc [1, 24*4*num_days]
num_days = 5
#discrepance = late - early, thuoc [mindis, maxdis]
min_discrepancy = 2*4
max_discrepancy = 24*4
#path toi file luu data
path_data = f'{n}points_{num_days}days_doChenh{max_discrepancy - min_discrepancy}costNho1_5.txt'
#delay thuoc [min_delay, max_delay]
min_delay = 1
max_delay = 8
#cost[i][j] thuoc [min_cost, max_cost]
min_cost = 1
max_cost = 5
#time[i][j] thuoc [min_time, max_time]
min_time = 1
max_time = 12

early = [0 for i in range(n+1)]
late = [0 for i in range(n+1)]
delay = [0 for i in range(n+1)]
cost = [[0 for i in range(n+1)] for j in range(n+1)]
time = [[0 for i in range(n+1)] for j in range(n+1)]

#thoi gian trong vong num_days ngay, don vi thoi gian 15' ~ gia tri dao dong tu 1 don vi --> 24*num_days*4 don vi thoi gian
early = np.random.uniform(1, 24*num_days*4, n+1).tolist()
early[0] = 0

def create_data():
  for i in range(1, n+1):
    early[i] = int(early[i])
    #early[i] = random.randint(1, 288)
    discrepancy = random.randint(min_discrepancy, max_discrepancy)
    late[i] = early[i] + discrepancy
    delay[i] = random.randint(min_delay, max_delay)

  for i in range(0, n+1):
    for j in range(0, n+1):
      if j != i:
        cost[i][j] = random.randint(min_cost, max_cost)
        #time di chuyen tu 15' den 4h
        time[i][j] = random.randint(min_time, max_time)

  '''
  for i in range(n+1):
    print(f'{i}: early {early[i]}, late {late[i]}, delay {delay[i]}')
  for i in range(n+1):
    for j in range(n+1):
      print(f'[{i}, {j}]: cost {cost[i][j]}; time {time[i][j]};', end = '')
    print('\n')
  '''
 
  with open(path_data,'w') as file:
    #dong 1 ghi gia tri n
    file.write(str(n) + '\n')
    #n + 1 dong tiep theo ghi gia tri early[i], late[i], delay[i]
    file.write('0 0 0\n')
    for i in range(1, n+1):
      file.write(f'{early[i]} {late[i]} {delay[i]}\n')
    #n+1 dong tiep theo ghi lai ma tran cost kich thuoc (n+1)*(n+1)
    for i in range(n+1):
      file.write(' '.join(str(cost[i][j]) for j in range(n+1)) + '\n')
    #n+1 dong tiep theo ghi lai ma tran time kich thuoc (n+1)*(n+1)
    for i in range(n+1):
      file.write(' '.join(str(time[i][j]) for j in range(n+1)) + '\n')
    file.write(f'\n\nearly[i] thuoc [1, {24*num_days*4}]\n')
    file.write(f'Do chenh giua early[i] va late[i] thuoc [{min_discrepancy}, {max_discrepancy}]\n')
    file.write(f'delay[i] thuoc [{min_delay}, {max_delay}]\n')
    file.write(f'cost[i][j] thuoc [{min_cost}, {max_cost}]\n')
    file.write(f'time[i][j] thuoc [{min_time}, {max_time}]\n')

create_data()