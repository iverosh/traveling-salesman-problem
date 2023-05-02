import random
import numpy as np


def generate(n, xy_range):
    cities = []
    for i in range(n):
        x = random.randint(0, xy_range)
        y = random.randint(0, xy_range)
        cities.append((x, y))
    distance = []
    for i in range(n):
        dist = []
        for j in range(n):
            if i == j:
                dist.append(0)
            else:
                dist.append(((cities[i][0] - cities[j][0]) ** 2 + (cities[i][1] - cities[j][1]) ** 2) ** 0.5)
        distance.append(dist)
    return cities, distance



