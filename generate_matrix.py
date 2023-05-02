import random
import numpy as np


def generate(n, xy_range, seed):
    random.seed = seed
    cities = []
    for i in range(n):
        x = random.randint(0, xy_range)
        y = random.randint(0, xy_range)
        cities.append((x, y))
    distance = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                distance[i][j] = 0
            else:
                distance[i][j] = ((cities[i][0] - cities[j][0]) ** 2 + (cities[i][1] - cities[j][1]) ** 2) ** 0.5
    return cities, distance



