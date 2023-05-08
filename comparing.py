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
for n_cities in range(6, 15):
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

        # bf_start = time.time()
        # solve_tsp_brute_force(distance_np)
        # bf_end = time.time()

        dynamic_start = time.time()
        path, dist = solve_tsp_dynamic_programming(distance_np)
        dynamic_end = time.time()

        dynamic_av_time += dynamic_end - dynamic_start
        ga_av_time += ga_end - ga_start
        # bf_av_time += bf_end - bf_start

        if ga_dist - dist < 0 and dist - ga_dist > 10 ** -10:
            print(ga_dist, dist)
            print(ga_sol)
            for j in path:
                print(cities[j], end=" ")
            plot_cities(cities, path)
            ga.plotRoute_out(ga_sol)

        print(i, ":", ga_dist - dist)
        if abs(ga_dist - dist) > 10 ** -10:  # вот тут хз, убирать ли этот иф
            delta_ += ga_dist - dist  #
            # print(delta_)

    dynamic_time.append(dynamic_av_time / i * 1000)
    bf_time.append(bf_av_time / i * 1000)
    ga_time.append(ga_av_time / i * 1000)
    delta.append(delta_ / i)
    cities_count.append(n_cities)

for i in range(1, len(ga_time)):


for i in range(len(cities_count)):
    print("Городов: {}, Среднее время GA: {} мс, Dynamic: {} мс, Brute Force: {} мс, Ошибка GA: {:.2}".format(
        cities_count[i],
        ga_time[i],
        dynamic_time[i],
        bf_time[i],
        delta[i]))
