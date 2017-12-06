from pyeasyga import pyeasyga
import random
import math
import copy

# Чтение файла
def read(file, lst):
    thread = open(file)
    for row in thread:
        lst.append(row.replace('\n','').split(' '))
    thread.close()
# Определение ценности елемента
def avgPersent(elem):
    k = 1/3
    return k*int(elem[0])+k*float(elem[1])-k*int(elem[2])

def maxFill(arr, i):
    for row in arr:
        row[i]=99999999


def create_individual(data):
    Gen = [0]*len(data)
    global worth
    tmp_index = random.randint(0,len(worth)-1)
    global ways
    copy_ways = copy.deepcopy(ways)
    global weight
    global volume
    checkArr = [tmp_index]
    maxFill(copy_ways, tmp_index)
    curr_weight = float(data[tmp_index][0])
    curr_volume = float(data[tmp_index][1])
    a = True
    while a: 
        tmp_row = copy_ways[tmp_index]
        min_index = tmp_row.index(min(tmp_row))
        curr_weight += float(data[min_index][0])
        curr_volume += float(data[min_index][1])
        
        if(weight < curr_weight or volume < curr_volume):
            a = False
        else:
            maxFill(copy_ways, min_index)
            checkArr.append(min_index)
            tmp_index = min_index
    for i in checkArr:
        Gen[i] = 1
    return Gen




def crossover(parent_1, parent_2):
    points = []
    while(len(points)<4):
        temp = random.randrange(1, len(parent_1))
        result=list(set(points) & set([temp]))
        if len(result) == 0:
            points.append(temp)
    points.sort()
    child_1 = parent_1[:points[0]] + parent_2[points[0]:points[1]] + parent_1[points[1]:points[2]] + parent_2[points[2]:]
    child_2 = parent_2[:points[0]] + parent_1[points[0]:points[1]] + parent_2[points[1]:points[2]] + parent_1[points[2]:]
    return child_1, child_2






def mutate(individual):
    points = []
    while(len(points)<4):
        temp = random.randrange(1, len(individual))
        result=list(set(points) & set([temp]))
        if len(result) == 0:
            points.append(temp)
    for mutate_index in points:
        if individual[mutate_index] == 0:
            individual[mutate_index] == 1
        else:
            individual[mutate_index] == 0


def selection(population):
    population.sort(key=lambda val: int(val.fitness), reverse = True)
    percent20 = int(len(population)* 0.7)
    tmp_population = population[:percent20]
    return tmp_population[random.randint(0, percent20-1)]


def fitness (individual, data):
    fitness = 0
    MaSS = 0
    Volum = 0
    global weight
    global volume
    for (selected, (mass, ngVol, cost)) in zip(individual, data):
        if selected:
            MaSS = float(mass)
            Volum = float(ngVol)
            fitness = float(cost) * 0.4/(0.0000398 * float(mass) + 0.05 * float(ngVol))
            #print([MaSS,Volum,cost],[1/(0.0000398*float(mass)),1/(0.05*float(ngVol)),0.4*float(cost)],fitness)
    #print(fitness, individual)
    if(volume < Volum or weight < MaSS):
        fitness = 1/(1+math.exp(0.03*fitness))
    
    return fitness





data = []
read('./26.txt',data)
weight = float(data[0][0]) #Максимальная масса
volume = float(data[0][1]) #Максимальный объём
data = data[1:]

worth = [] # Таблица ценностей (какие вещи наиболее ценны)

ways = [] # Таблица путей сделанных по разности ценности элементов
for elem in data:
    worth.append(avgPersent(elem))
a = min(worth) + 1
for i in range(0,len(worth)):
    worth[i] -= a

# Построение таблицы растояния оператора Кроссинговера
for elem in worth:
    row = []
    for item in worth:
        if(elem == item):
            row.append(9999999999)
        else:
            row.append(abs(elem - item))
    ways.append(row)

ga = pyeasyga.GeneticAlgorithm(data,
                               population_size=100,
                               generations=100,
                               crossover_probability=0.8,
                               mutation_probability=0.2,
                               elitism=True,
                               maximise_fitness=True)

# and set the Genetic Algorithm's ``mutate_function`` attribute to
# your defined function
ga.mutate_function = mutate

# and set the Genetic Algorithm's ``crossover_function`` attribute to
# your defined function
ga.crossover_function = crossover

# and set the Genetic Algorithm's ``create_individual`` attribute to
# your defined function
ga.create_individual = create_individual

# and set the Genetic Algorithm's ``selection_function`` attribute to
# your defined function
ga.selection_function = selection

# and set the Genetic Algorithm's ``fitness_function`` attribute to
# your defined function
ga.fitness_function = fitness


ga.run()

print(ga.best_individual())
mass = 0
V = 0
cost = 0
ans = ga.best_individual()

for (select, (profit)) in zip(ans[1], data):
        if select:
            mass += float(profit[0])
            V += float(profit[1])
            cost += float(profit[2])


print('Mass: '+str(weight))
print('V: '+str(volume))
print('Mass_ans: '+str(mass))
print('V_ans: '+str(V))
print('Cost_ans: '+str(cost))









