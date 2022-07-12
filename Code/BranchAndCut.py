
import os
import copy

path_data = '50points_5days_doChenh88costNho1_5.txt'
path_result = 'BAC_result_of_' + path_data

if os.path.exists(path_result):
    os.remove(path_result)

def get_data(path):
    with open(path, 'r') as file:
        n = int(file.readline())
        cost = []
        time = []
        points = {}
        for i in range(n+1):
            e, l, d = file.readline().strip().split(' ')
            points[i] = (int(e), int(l), int(d))
        for i in range(n+1):
            costi = [int(a) for a in file.readline().strip().split(' ')]
            cost.append(costi)
        for i in range(n+1):
            timei = [int(a) for a in file.readline().strip().split(' ')]
            time.append(timei)
    
    return n, points, cost, time

def sort_points(points):
    sorted_points = [point[0] for point in sorted(points.items(), key = lambda x: x[1])]
    return sorted_points

n, points, cost, time = get_data(path_data)
sorted_point = sort_points(points)
result = [-1 for i in range(n+1)]
best_result = []
visited = [0 for i in range(n+1)]
best_total_cost = 100000000

cost_min = cost[0][1]
for i in range(n+1):
    for j in range(n+1):
        if i != j:
            cost_min = min(cost_min, cost[i][j])


#for point in sorted_point:
#    print(f'{point}: {points[point]}, ', end='')


'''
cur_idx la vi tri dang xet cua loi giai result
cur_time la thoi gian khi dang dung o result[cur_idx] (tinh ca delay[cur_idx]) va chuan bi xuat phat toi diem tiep theo
cur_total_cost la tong quang duong cho toi thoi diem hien tai
'''
def back_tracking(cur_idx, cur_time, cur_total_cost):
    #neu da tao ra loi giai day du
    if (cur_idx == n):
        global best_total_cost
        global best_result

        total_cost = cur_total_cost + cost[result[cur_idx]][0]
        if (best_total_cost > total_cost):
            best_total_cost = total_cost
            best_result = copy.deepcopy(result)

        print('Loi giai: ', result)
        print('Tong quang duong cua loi giai: ' + str(total_cost))
        print('Best total cost hien tai: ' + str(best_total_cost))

        with open(path_result, 'a') as file:
            str1 = ' '.join(str(result[i]) for i in range(n+1))
            file.write('+++++++++++++++++++++++++++\n')
            file.write('Loi giai: [' + str1 + ']\n')
            file.write('Tong quang duong cua loi giai: ' + str(total_cost) + '\n')
            file.write('Best total cost hien tai: ' + str(best_total_cost) + '\n')
         
        return
    
    cur_point = result[cur_idx]
    #kiem tra dieu kien cat nhanh
    '''c = []
    for i in range(n+1):
        for j in range(n+1):
            if visited[i] + visited[j] == 0:
                c.append(cost[i][j])
    tmp = cost[cur_point][1]
    for i in range(2, n+1):
        tmp = min(tmp, cost[cur_point][i])
    c.append(tmp)
    c = sorted(c)
    if cur_total_cost + sum(c[0 : n - cur_idx + 1])> best_total_cost:
        return'''
    
    if cur_total_cost + (n - cur_idx + 1)*cost_min > best_total_cost:
        return 
    #kiem tra xem neu dang o cur_point thi co lam cho poin nao qua thoi gian ko
    for point in sorted_point:
        if (visited[point] == 0 and cur_time + time[cur_point][point] > points[point][1]):
            return

    #duyet qua tung point chua tham de dat vao vi tri cur_idx + 1
    for point in sorted_point:
        if visited[point]:
            continue
        result[cur_idx + 1] = point
        visited[point] = 1
        #den som thi cho``, den muon hon early[point] thi tinh thoi gian den
        new_cur_time = max(cur_time + time[cur_point][point] + points[point][2], points[point][0] + points[point][2])
        new_cur_total_cost = cur_total_cost + cost[cur_point][point]
        back_tracking(cur_idx + 1, new_cur_time, new_cur_total_cost)
        visited[point] = 0 

def solve():
    '''if os.path.exists(path_result):
        os.remove(path_result)'''
    result[0] = 0
    visited[0] = 1
    back_tracking(0, 0, 0)
    if best_total_cost == 100000000:
        print('The problem does not have an optimal solution!')
        with open(path_result, 'a') as file:
            file.write('The problem does not have an optimal solution!')
        return -1
    else:
        print('best result: ', best_result)
        #for p in best_result:
        #    print(f'{p}: {points[p]}')
        print('best cost: ', best_total_cost)
        with open(path_result, 'a') as file:
            str1 = ' '.join(str(best_result[i]) for i in range(n+1))
            file.write('+++++++++++++++++++++++++++\n')
            file.write('Loi giai tot nhat: [' + str1 + ']\n')
            file.write('Best total cost: ' + str(best_total_cost) + '\n')
        return 1


if __name__ == "__main__":
    solve()

    
