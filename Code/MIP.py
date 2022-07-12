from ortools.linear_solver import pywraplp

path_data = '50points_3days_doChenh88costNho.txt'
path_result = 'MIP_result_of_' + path_data

def get_data(path):
    with open(path, 'r') as file:
        n = int(file.readline())
        early = []
        late = []
        delay = []
        cost = []
        time = []
        
        for i in range(n+1):
            e, l, d = file.readline().strip().split(' ')
            early.append(int(e))
            late.append(int(l))
            delay.append(int(d))
        for i in range(n+1):
            costi = [int(a) for a in file.readline().strip().split(' ')]
            cost.append(costi)
        for i in range(n+1):
            timei = [int(a) for a in file.readline().strip().split(' ')]
            time.append(timei)
    
    return n, early, late, delay, cost, time

def mip_solve():
    #lay du lieu
    n, early, late, delay, cost, time = get_data(path_data)

    #tao MIP solver voi SCIP backend
    solver = pywraplp.Solver.CreateSolver('SCIP')

    '''
    TAO BIEN
    '''
    #x[i][j] = 1 neu co duong di tu i den j
    x = {}
    for i in range(n+1):
        for j in range(n+1):
            x[i, j] = solver.IntVar(0, 1, 'x['+str(i)+','+str(j)+']')
    #y[i] = k neu thoi diem den tham diem i la k (tinh tu moc 0)
    y = {}
    for i in range(n+1):
        y[i] = solver.IntVar(early[i], late[i], 'y['+str(i)+']')

    '''
    RANG BUOC
    '''
    #moi dia diem chi di qua 1 lan
    for i in range(n+1):
        cons = solver.Constraint(1, 1)
        for j in range(n+1):
            cons.SetCoefficient(x[i, j], 1)
    for i in range(n+1):
        cons = solver.Constraint(1, 1)
        for j in range(n+1):
            cons.SetCoefficient(x[j, i], 1)

    #cac dia diem ko di qua chinh no
    for i in range(n+1):
        cons = solver.Constraint(0, 0)
        cons.SetCoefficient(x[i, i], 1)

    #nguoi giao hang xuat phat tu 0
    cons = solver.Constraint(0, 0)
    cons.SetCoefficient(y[0], 1)

    #neu co duong di tu i den j thi thoi gian giao hang toi j phai thoa man dieu kien
    maxtime = -1
    for i in range(n+1):
        for j in range(n+1):
            maxtime = max(maxtime, time[i][j])
    a = max(delay) + maxtime + max(late)
    for i in range(n+1):
        for j in range(1, n+1):
            solver.Add(y[i] - y[j] + a*x[i, j] <= a - delay[i] - time[i][j])

    #tao ham muc tieu
    objective_terms = []
    for i in range(n+1):
        for j in range(n+1):
            objective_terms.append(cost[i][j]*x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    '''
    IN LOI GIAI
    '''
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("Solution:")
        print('Objective value = ', solver.Objective().Value())
        #in ra chu trinh
        cur = 0
        tour = []
        for i in range(n+1):
            tour.append(cur)
            for next in range(n+1):
                if x[cur, next].solution_value() != 0:
                    cur = next
                    break
        print('Chu trinh: ', tour)
        #in ra thoi gian tham cac diem i
        time_visit = []
        for i in range(n+1):
            time_visit.append((y[i].name(), y[i].solution_value()))
        time_visit = sorted(time_visit, key = lambda x: x[1])
        print('Time visit: ', time_visit)

        with open(path_result, 'w') as file:
            file.write(f'Objective value =  {solver.Objective().Value()}\n')
            str1 = ' '.join(str(tour[i]) for i in range(n+1))
            file.write(f'Tour: ' + str1 + '\n')
            str2 = ' '.join(f'({t[0]}, {t[1]})' for t in time_visit)
            file.write(f'Time visit: ' + str2 + '\n')
    else:
        print('the problem does not have an optimal solution')

if __name__ == '__main__':
    mip_solve()
        