import numpy as np
import matplotlib.pyplot as plt
import random
import math
import warnings
import numpy
import generate_matrix
import time
from BnB import plot_cities
from python_tsp.exact import solve_tsp_dynamic_programming, solve_tsp_brute_force
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

# hyperparameters
POP_SIZE = 100
N_ITER = 100
ELISTISM_FAC = 20


# coding the problem
class travelingSalesman(object):
    def __init__(self, n_cities=25, xy_range=2000, cities=[]):
        self.n_cities = n_cities
        self.xy_range = xy_range
        self.cities = cities
        if len(cities) == 0:
            for i in range(self.n_cities):
                x = random.randint(0, self.xy_range)
                y = random.randint(0, self.xy_range)
                cities.append((x, y))

    def plotCities(self):
        for city in self.cities:
            plt.plot(city[0], city[1], 'bo')

    def plotRoute(self, route):
        i = 0
        while i < len(route):
            if i + 1 == len(route):
                x_values = [route[i][0], route[0][0]]
                y_values = [route[i][1], route[0][1]]
                plt.plot(x_values, y_values)
            else:
                x_values = [route[i][0], route[i + 1][0]]
                y_values = [route[i][1], route[i + 1][1]]
                plt.plot(x_values, y_values)
            i += 1

    def generateRoute(self):
        route = self.cities
        random.shuffle(route)
        return route

    def getDistance(self, route):
        distance = 0
        i = 0
        while i < len(route):
            if i + 1 == len(route):
                distance += math.sqrt(((route[i][0] - route[0][0]) ** 2) + ((route[i][1] - route[0][1]) ** 2))
            else:
                distance += math.sqrt(((route[i][0] - route[i + 1][0]) ** 2) + ((route[i][1] - route[i + 1][1]) ** 2))
            i += 1

        return distance

    def computeFitness(self, route):
        return 100000 / float(self.getDistance(route))


def crossover(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def selectParents(population):
    prob_dist = []
    fit_sum = sum(population['fitness'])
    for i in population['fitness']:
        prob_dist.append(i / fit_sum)

    done = False
    p1_idx = 0
    p2_idx = 0
    while not done:
        if p1_idx == p2_idx:
            p1_idx = np.random.choice(np.arange(0, 100), p=prob_dist)
            p2_idx = np.random.choice(np.arange(0, 100), p=prob_dist)
        else:
            done = True

    parent1 = population.iloc[p1_idx]['solutions']
    parent2 = population.iloc[p2_idx]['solutions']

    return parent1, parent2


def plotCities_out(cities):
    for city in cities:
        plt.plot(city[0], city[1], 'bo')


def plotRoute_out(route):
    i = 0
    while i < len(route):
        if i + 1 == len(route):
            x_values = [route[i][0], route[0][0]]
            y_values = [route[i][1], route[0][1]]
            plt.plot(x_values, y_values, "g--")
        else:
            x_values = [route[i][0], route[i + 1][0]]
            y_values = [route[i][1], route[i + 1][1]]
            plt.plot(x_values, y_values, "g--")
        i += 1
    plt.show()


def startMethod(cities_input, n_cities_input, n_iter):
    ts = travelingSalesman(n_cities=n_cities_input, cities=cities_input)

    # baseline for our model
    random_route = ts.generateRoute()
    # ts.plotCities()
    # ts.plotRoute(random_route)
    # plt.show()
    # print('initial route distance is: ' + str(ts.getDistance(random_route)))

    # initialize population
    population = pd.DataFrame({'solutions': [], 'fitness': []})
    for _ in range(POP_SIZE):
        route = ts.generateRoute()
        population = population.append({'solutions': route, 'fitness': ts.computeFitness(route)}, ignore_index=True)
        # population = pd.concat([population, pd.DataFrame({'solutions': route, 'fitness': ts.computeFitness(route)})], ignore_index=True)

    population = population.sort_values('fitness', ascending=False, ignore_index=True)

    for _ in range(n_iter):
        evolved_pop = []
        # perform elitism
        for i in range(ELISTISM_FAC):
            evolved_pop.append(population.iloc[i]['solutions'])

        while len(evolved_pop) < POP_SIZE:
            # select parents
            parent1, parent2 = selectParents(population)
            # choose random crossover point & perform crossover
            evolved_pop.append(crossover(parent1, parent2))

        # set the evolved_pop as the new population for the next generation
        population['solutions'] = evolved_pop
        for index, row in population.iterrows():
            population.at[index, 'fitness'] = ts.computeFitness(row['solutions'])

        population = population.sort_values('fitness', ascending=False, ignore_index=True)

    population = population.sort_values('fitness', ascending=False, ignore_index=True)
    solution = population.iloc[0]['solutions']
    # ts.plotCities()
    # ts.plotRoute(solution)
    # plt.show()
    # print('final route distance is: ' + str(ts.getDistance(solution)))
    return solution, ts.getDistance(solution)


n_iters = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
n_cities = 19
xy_range = 100
ga_time = []
delta_ = 0
ga_av_time = 0
delta = []

for i in n_iters:
    # for j in range(10):
    #     cities = cities_const
    #     ga_start = time.time()
    #     ga_sol, ga_dist = startMethod(cities, n_cities, i)
    #     ga_end = time.time()
    #     ga_av_time += ga_end - ga_start
    #     if abs(ga_dist - dist) > 10 ** -10:
    #         delta_ += (ga_dist - dist) / dist * 100
    cities, distance = generate_matrix.generate(n_cities, xy_range)
    distance_np = numpy.array(distance)
    path, dist = solve_tsp_dynamic_programming(distance_np)

    ga_start = time.time()
    ga_sol, ga_dist = startMethod(cities, n_cities, i)
    ga_end = time.time()
    ga_av_time = ga_end - ga_start

    delta_ = (ga_dist - dist) / dist * 100

    ga_time.append(ga_av_time * 1000)
    print(ga_dist, dist)
    print(i, ":", ga_dist - dist)
    delta.append(delta_)
print(delta)
print(ga_time)

'''
    График зависимости ошибки от количества итераций для ГА
                                                                '''
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)

ax1.plot(n_iters, delta, label='Genetic algorithm', linewidth=3)
ax1.grid()
ax1.set_xlabel('number of iters')  # Add an x-label to the axes.
ax1.set_ylabel('err, %')  # Add a y-label to the axes.
ax1.set_title("    График зависимости ошибки от количества итераций для ГА.")  # Add a title to the axes.
ax1.legend()
plt.show()
