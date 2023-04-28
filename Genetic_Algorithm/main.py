import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pandas as pd

# hyperparameters
POP_SIZE = 100
N_ITER = 500
ELISTISM_FAC = 20


# coding the problem
class travelingSalesman(object):
    def __init__(self, n_cities=25, xy_range=2000, cities=[]):
        self.n_cities = n_cities
        self.xy_range = xy_range
        self.cities = cities
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


if __name__ == '__main__':
    ts = travelingSalesman()

    # baseline for our model
    random_route = ts.generateRoute()
    ts.plotCities()
    ts.plotRoute(random_route)
    plt.show()
    print('initial route distance is: ' + str(ts.getDistance(random_route)))

    # initialize population
    population = pd.DataFrame({'solutions': [], 'fitness': []})
    for _ in range(POP_SIZE):
        route = ts.generateRoute()
        population = population.append({'solutions': route, 'fitness': ts.computeFitness(route)}, ignore_index=True)
        # population = pd.concat([population, pd.DataFrame({'solutions': route, 'fitness': ts.computeFitness(route)})], ignore_index=True)

    population = population.sort_values('fitness', ascending=False, ignore_index=True)

    for _ in range(N_ITER):
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
    ts.plotCities()
    ts.plotRoute(solution)
    plt.show()
    print('final route distance is: ' + str(ts.getDistance(solution)))
