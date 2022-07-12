from unicodedata import name
from ortools.sat.python import cp_model

path_data = '50points_3days_doChenh88costNho.txt'
path_result = 'CP_result_of_' + path_data
num_part = 200

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
    data = {}
    data['n'] = n
    data['early'] = early
    data['late'] = late
    data['delay'] = delay
    data['cost'] = cost
    data['time'] = time
    return data

#tra ve true neu co loi giai trong khoang [lb, ub]
def check_with_bound(lb, ub, data):
    #lay du lieu
    n, early, late, delay, cost, time = data['n'], data['early'], data['late'], data['delay'], data['cost'], data['time']
    '''c = []
    for i in range(m+1):
        for j in range(n+1):
            c.append(cost[i][j])
    c = sorted(c)
    
    min_lb = sum(c[0 : n+1])
    lb = min_lb
    ub = min_lb
    max_ub = sum(c[-n-1:])'''
    
    #tao CP-SAT model
    model = cp_model.CpModel()

    '''
    TAO BIEN
    '''
    #x[i, j] = 1 neu di tu i den j, = 0 neu nguoc lai
    x = {}
    for i in range(n+1):
        for j in range(n+1):
            x[i, j] = model.NewIntVar(0, 1, f'x[{i},{j}]')
    #y[i] the hien thu tu cua diem den i trong hanh trinh, bat dau tu 0
    '''y = {}
    for i in range(n+1):
        y[i] = model.NewIntVar(0, n, f'y[{i}]')'''
    #z[i] the hien thoi gian den diem i trong hanh trinh
    z = {}
    for i in range(n+1):
        z[i] = model.NewIntVar(early[i], late[i], f'z[{i}]')
    
    '''
    RANG BUOC
    '''
    #moi dia diem chi di qua 1 lan
    for i in range(n+1):
        model.AddExactlyOne(x[i,j] for j in range(n+1))
    for i in range(n+1):
        model.AddExactlyOne(x[j,i] for j in range(n+1))
    #cac dia diem ko di qua chinh no
    for i in range(n+1):
        model.Add(x[i,i] == 0)
    #nguoi giao hang xuat phat tu 0
    model.Add(z[0] == 0)
    #neu co duong di tu i den j thi thoi gian giao hang toi j phai thoa man dieu kien
    '''maxtime = -1
    for i in range(n+1):
        for j in range(n+1):
            maxtime = max(maxtime, time[i][j])
    a = max(delay) + maxtime + max(late)
    for i in range(n+1):
        for j in range(1, n+1):
            model.Add(z[i] - z[j] + a*x[i, j] <= a - delay[i] - time[i][j])'''
    #x[i, j] = 1 --> y[j] > y[i], voi j>0
    '''for i in range(n+1):
        for j in range(1, n+1):
            b = model.NewBoolVar('b')
            model.Add(x[i, j] == 1).OnlyEnforceIf(b)
            model.Add(x[i, j] != 1).OnlyEnforceIf(b.Not())
            model.Add(y[j] - y[i] == 1).OnlyEnforceIf(b)'''
    #x[i, j] = 1 --> z[j] - z[i] >= time[i, j] + delay[i], voi j > 0
    for i in range(n+1):
        for j in range(1, n+1):
            b = model.NewBoolVar('b')
            model.Add(x[i, j] == 1).OnlyEnforceIf(b)
            model.Add(x[i, j] != 1).OnlyEnforceIf(b.Not())
            model.Add(z[j] - z[i] >= time[i][j] + delay[i]).OnlyEnforceIf(b)
    #rang buoc alldiffent
    #model.AddAllDifferent(y[i] for i in range(n + 1))
    model.AddAllDifferent(z[i] for i in range(n + 1))
    #lb< tong quang duong < ub
    objective_terms = []
    for i in range(n + 1):
        for j in range(n + 1):
            objective_terms.append(cost[i][j]*x[i, j])
    model.AddLinearConstraint(cp_model.LinearExpr.Sum(objective_terms), lb, ub)

    '''
    TIM LOI GIAI
    '''
    #tao ham muc tieu
    #model.Minimize(cp_model.LinearExpr.Sum(objective_terms))
    #tao solver và giai model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    '''
    KIEM TRA LOI GIAI
    '''
    print('Kiem tra cho khoang [lb, ub] = ', lb, ',', ub)
    #in loi giai
    if status == cp_model.OPTIMAL:
        print("Solution:")
        #print('Objective value = ', solver.ObjectiveValue())
        for i in range(n):
            for j in range(n):
                if solver.Value(x[i, j]):
                    print(f'{x[i, j].Name()} = {solver.Value(x[i, j])}')
        '''for i in range(n):
            print(f'{y[i].Name()} =  {solver.Value(y[i])}')'''
        for i in range(n):
            print(f'{z[i].Name()} =  {solver.Value(z[i])}')
        return True
    else:
        print('The problem does not have an optimal solution in this bound!')
        return False

def solve():
    data = get_data(path_data)

    c = []
    for i in range(data['n']+1):
        for j in range(data['n']+1):
            if i != j:
                c.append(data['cost'][i][j])
    c = sorted(c)
    
    min_lb = sum(c[0 : data['n']+1])
    max_ub = sum(c[-data['n']-1:])
    print('Tim nghiem trong khoang: [', min_lb, ',', max_ub, ']')

    offset = (max_ub-min_lb)//num_part + 1
    for i in range(num_part):
        lb = min_lb + i*offset
        ub = min_lb + (i+1)*offset
        if check_with_bound(lb, ub, data):
            break

if __name__ == '__main__':
    solve()

