import Genetic_Algorithm as ga
import numpy
import generate_matrix
import time
from BnB import plot_cities
from python_tsp.exact import solve_tsp_dynamic_programming, solve_tsp_brute_force

xy_range = 100

bf_time = []
dynamic_time = []
ga_time = []
delta = []
cities_count = []
for n_cities in range(6, 35):
    print("Города:", n_cities)
    dynamic_av_time = 0
    bf_av_time = 0
    ga_av_time = 0
    delta_ = 0
    for i in range(1, 10):
        cities, distance = generate_matrix.generate(n_cities, xy_range)
        distance_np = numpy.array(distance)
        ga_start = time.time()
        ga_sol, ga_dist = ga.startMethod(cities, n_cities)
        ga_end = time.time()
        ga_av_time += ga_end - ga_start
        print(i, ":", ga_dist)
    ga_time.append(ga_av_time / i * 1000)
    cities_count.append(n_cities)

for i in range(len(cities_count)):
    print("Городов: {}, Среднее время GA: {} мс".format(
        cities_count[i],
        ga_time[i],)
    )