import numpy as np
import random as rd


PENALTY = 1000000


'''
LAY DU LIEU TU PATH_DATA
'''
def get_data():
    path_data = '50points_5days.txt'
    path_result = 'CP_result_of_50points_5days.txt'
    with open(path_data, 'r') as file: 
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


'''
DINH NGHIA LOP CA THE
'''
class Individual:
    tour = None # Thu tu tham cac vi tri
    cost_of_tour = None # Tong chi phi di chuyen
    is_valid = None # Co thoa man dieu kien hay khong
    invalid_coef = None # He so vi pham rang buoc
    def __init__(self, tour = [], cost_of_tour = -1, is_valid = False, invalid_coef = 0):
        self.tour = tour
        self.cost_of_tour = cost_of_tour
        self.is_valid = is_valid
        self.invalid_coef = invalid_coef


'''
DINH NGHIA LOP QUAN THE
'''
class Population:

    pop_size = None # Kich thuoc quan the
    cities_num = None # So vi tri phai den
    iteration_num = None # So vong lap
    cost = None # Chi phi di chuyen
    time = None # Thoi gian di chuyen
    early = None # Thoi gian som nhat nhan hang
    late = None # Thoi gian muon nhat nhan hang
    delay = None # Thoi gian phai dung lai 

    popu = None # Cac ca the hien tai cua quan the
    best_indivi = None # Luu dap an tot nhat
    max_invalid_coef = None # Gia tri phat lon nhat

    def __init__(self, pop_size, iteration_num, cities_num, cost, time, early, late, delay):
        self.pop_size = pop_size
        self.iteration_num = iteration_num
        self.cities_num = cities_num
        self.cost = cost
        self.time = time
        self.early = early
        self.late = late
        self.delay = delay

    ###
    # CAC HAM HO TRO TINH TOAN
    ###

    # Tinh tong chi phi di chuyen cua mot cach di chuyen
    def get_cost(self, tour):
        cost = 0
        pre_city = tour[-1]
        for city in tour:
            cost += self.cost[pre_city][city]
            pre_city = city
        return cost

    # Tinh he so vi pham rang buoc
    def get_invalid_coefficient(self, tour):
        cur_time = 0
        coef = 0
        pre_city = tour[0]
        for city in tour[1:]:
            cur_time += self.time[pre_city][city]
            if cur_time > self.late[city]: 
                #coef += (cur_time - self.late[city])**2
                coef += 1
                cur_time = self.late[city]
            else:
                cur_time = max(cur_time, self.early[city])
            cur_time += self.delay[city]       
            pre_city = city
        return coef     

    # Kiem tra mot cach di chuyen vi pham rang buoc
    def is_valid_tour(self, tour):
        cur_time = 0
        pre_city = tour[0]
        for city in tour[1:]:
            cur_time += self.time[pre_city][city]
            if cur_time > self.late[city]: 
                #print(str(cur_time) + '_' + str(pre_city) + '_' + str(city))
                return False
            cur_time = max(cur_time, self.early[city])
            cur_time += self.delay[city]       
            pre_city = city
        return True

    def get_valid_random_tour(self):
        while (True):
            tour = np.concatenate([[0], np.random.permutation(np.arange(1, self.cities_num)), [0]]).tolist()
            if (self.is_valid_tour(tour) == True):
                print("\n")
                print(tour)
                print("\n") 
                return tour

    # Gia tri tong ket qua mot dap an
    def comp(self, indivi):
        x1 = indivi.cost_of_tour
        #x2 = int((invidi.invalid_coef/self.max_invalid_coef) * PENALTY)
        x2 = indivi.invalid_coef * PENALTY
        return x1 + x2

    # Chuyen tu cach di chuyen thanh mot doi tuong di chuyen
    def tour_to_individual(self, tour):
        cost = self.get_cost(tour)
        is_valid = self.is_valid_tour(tour)
        invalid_coef = 0
        if (is_valid == False):
            invalid_coef = self.get_invalid_coefficient(tour)
            self.max_invalid_coef = max(self.max_invalid_coef, invalid_coef)
        return Individual(tour, cost, is_valid, invalid_coef)



    ###
    # CAC PHA CUA THUAT TOAN
    ###

    # Khoi tao quan the
    def initialization(self):
        self.popu = []
        self.max_invalid_coef = 0
        for i in range(self.pop_size):
            tour = np.concatenate([[0], np.random.permutation(np.arange(1, self.cities_num))]).tolist()
            self.popu.append(self.tour_to_individual(tour)) 

        self.popu.sort(key=self.comp)

    # Lai ghep hai ca the
    def crossover(self, p1, p2):
        n = self.cities_num
        first = rd.randint(1, n - 1) # Vi tri dau tien phai luon la vi tri 0
        second = rd.randint(first + 1, n)
        c1 = np.copy(p1)
        c2 = np.copy(p2)
        c1[first:second] = np.zeros(self.cities_num)[first:second]
        c2[first:second] = np.zeros(self.cities_num)[first:second]
        it = first
        for i in range(1,n):
            x = p2[i]
            if not(x in c1):
                c1[it] = x
                it+=1
        it = first
        for i in range(1,n):
            x = p1[i]
            if not(x in c2):
                c2[it] = x
                it+=1
        return c1, c2

    # Dot bien ca the
    def mutate(self, tour):
        cur_time = 0
        for i in range(1,len(tour)):
            cur_time += self.time[tour[i-1]][tour[i]]
            if cur_time > self.late[tour[i]]: 

                if i==1 :
                    t = rd.randint(i+1,self.cities_num-1)
                    tour[i], tour[t] = tour[t], tour[i]
                    return tour

                if rd.randint(0,3) > 0:
                    t = rd.randint(1,i-1)
                    u = tour[i]
                    tour[t+1:i+1] = tour[t:i]
                    tour[t] = u
                else:
                    rd.shuffle(tour[1:i])
                return tour

            else:
                cur_time = max(cur_time, self.early[tour[i]])
            cur_time += self.delay[tour[i]]    

        [t1,t2] = rd.sample(list(range(1,self.cities_num)),2)
        tour[t1], tour[t2] = tour[t2], tour[t1]  
        return tour

    # Bat dau qua trinh tien hoa
    def evolution(self):
        for iter in range(self.iteration_num):
            # Chon loc tu nhien
            p1 = int(self.pop_size * 0.3) # Lay ra 30% ca the tot nhat
            self.popu = self.popu[:p1]
            good_indivi = self.popu.copy()

            # Khoi phuc quan the
            while (len(self.popu) < self.pop_size*0.5):
                tour = np.concatenate([[0], np.random.permutation(np.arange(1, self.cities_num)), [0]])
                #print(tour)
                self.popu.append(self.tour_to_individual(tour)) 

            while (len(self.popu) < self.pop_size):
                [parent1, parent2] = rd.sample(good_indivi,2)
                tour1, tour2 = self.crossover(parent1.tour, parent2.tour)
                if (rd.randint(1,100) < 40):
                    tour1 = self.mutate(tour1)
                    tour2 = self.mutate(tour2)
                self.popu.append(self.tour_to_individual(tour1))
                self.popu.append(self.tour_to_individual(tour2))

                # Dot bien
                tour = rd.sample(good_indivi,1)[0].tour
                if (rd.randint(0,101) < 40):
                    #print(tour)
                    tour = self.mutate(tour)
                    #print(tour)
                    self.popu.append(self.tour_to_individual(tour)) 



            self.popu.sort(key=self.comp)

            #print(self.popu[0].tour)
            #print(self.popu[0].invalid_coef)
            #print(self.comp(self.popu[0]))
            #print('\n')

        self.best_indivi = self.popu[0]
        #for invidi in self.popu:            
        #    if invidi.is_valid == True:
        #        if invidi.cost_of_tour < self.best_invidi.cost_of_tour:
        #            self.best_invidi = invidi
        
            


def main():
    data = get_data()

    for i in range(100):
        ga_population = Population(
            pop_size = 2000, 
            iteration_num = 1000,
            cities_num = data['n'] + 1,
            cost = data['cost'],
            time = data['time'],
            early = data['early'],
            late = data['late'],
            delay = data['delay'])

        ga_population.initialization()
        ga_population.evolution()
        best_indivi = ga_population.best_indivi
        print(best_indivi.tour)
        print(ga_population.comp(best_indivi))


 
if __name__ == "__main__":
    main()

