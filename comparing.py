import Genetic_Algorithm as ga
import numpy
import generate_matrix
import time
from BnB import BnB, plot_cities
from python_tsp.exact import solve_tsp_dynamic_programming
n_cities = 5
xy_range = 100
#bnb_time = []
dynamic_time = []
ga_time = []
delta = []
cities_count = []
for n_cities in range(4, 20, 2):
    print("Города:", n_cities)
    dynamic_av_time = 0
    # bnb_av_time = 0
    ga_av_time = 0
    delta_ = 0
    for i in range(1, 4):
        cities, distance = generate_matrix.generate(n_cities, xy_range)
        #print(cities)
        ga_start = time.time()
        ga_sol, ga_dist = ga.startMethod(cities, n_cities)
        ga_end = time.time()

        # bnb_start = time.time()
        # bnb_sol, bnb_dist = BnB(distance)
        # bnb_end = time.time()
        # bnb_av_time += bnb_end - bnb_start

        dynamic_start = time.time()
        path, dist = solve_tsp_dynamic_programming(numpy.array(distance))
        dynamic_end = time.time()

        dynamic_av_time += dynamic_end - dynamic_start
        ga_av_time += ga_end - ga_start

        if ga_dist - dist < 0 and dist - ga_dist > 10 ** -10:
            print(ga_dist, dist)
            print(ga_sol)
            for j in path:
                print(cities[j], end=" ")
            plot_cities(cities, path)
            ga.plotRoute_out(ga_sol)

        print(i, ":", ga_dist - dist)
        if abs(ga_dist - dist) > 10 ** -10:    # вот тут хз, убирать ли этот иф
            delta_ =+ ga_dist - dist           #
            #print(delta_)

    #bnb_time.append(bnb_av_time / i * 1000)
    dynamic_time.append(dynamic_av_time / i * 1000)
    ga_time.append(ga_av_time / i * 1000)
    delta.append(delta_ / i)
    cities_count.append(n_cities)



for i in range(len(cities_count)):
    print("Городов: {}, Среднее время GA: {:.2} мс, Dynamic: {:.2} мс, Ошибка GA: {:.2}".format(cities_count[i],
                                                                                               ga_time[i],
                                                                                               dynamic_time[i],
                                                                                               delta[i]))




