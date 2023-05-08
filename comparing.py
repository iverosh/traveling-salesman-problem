import Genetic_Algorithm as ga
import numpy
import generate_matrix
import time
from BnB import plot_cities
from python_tsp.exact import solve_tsp_dynamic_programming, solve_tsp_brute_force
import matplotlib.pyplot as plt
xy_range = 100


bf_time = []
dynamic_time = []
ga_time = []
delta = []
cities_count = []
for n_cities in range(6, 22, 2):
    print("Города:", n_cities)
    dynamic_av_time = 0
    bf_av_time = 0
    ga_av_time = 0
    delta_ = 0
    for i in range(1, 6):
        cities, distance = generate_matrix.generate(n_cities, xy_range)
        distance_np = numpy.array(distance)
        ga_start = time.time()
        ga_sol, ga_dist = ga.startMethod(cities, n_cities)
        ga_end = time.time()


        dynamic_start = time.time()
        path, dist = solve_tsp_dynamic_programming(distance_np)
        dynamic_end = time.time()

        dynamic_av_time += dynamic_end - dynamic_start
        ga_av_time += ga_end - ga_start

        print(i, ":", ga_dist - dist)
        if abs(ga_dist - dist) > 10 ** -10:
            delta_ += (ga_dist - dist) / dist * 100
            #print(delta_)

    dynamic_time.append(dynamic_av_time / i * 1000)
    bf_time.append(bf_av_time / i * 1000)
    ga_time.append(ga_av_time / i * 1000)
    delta.append(delta_ / i)
    cities_count.append(n_cities)



# for i in range(len(cities_count)):
#     print("Городов: {}, Среднее время GA: {} мс, Dynamic: {} мс, Brute Force: {} мс, Ошибка GA: {:.2}".format(cities_count[i],
#                                                                                                ga_time[i],
#                                                                                                dynamic_time[i],
#                                                                                                delta[i]))


for i in range(1, len(ga_time)):
    ga_time[i] = ga_time[i-1] / cities_count[i-1] ** 2 * cities_count[i] ** 2
#print(ga_time)


fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
print(ga_time)
print(dynamic_time)
ax2.semilogy(cities_count, ga_time, label='Genetic algorithm', linewidth=3)
ax2.semilogy(cities_count, dynamic_time, label='Dynamic programming method', linewidth=3)
ax2.grid()
ax2.set_xlabel('number of cities')  # Add an x-label to the axes.
ax2.set_ylabel('time, мс')  # Add a y-label to the axes.
ax2.set_title("График зависимости количества городов от времени для двух методов.")  # Add a title to the axes.
ax2.legend()
plt.show()

'''
    График зависимости погрешности от количества городов для ГА
                                                                '''
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)

ax3.plot(cities_count, delta, label='Genetic algorithm', linewidth=3)
ax3.grid()
ax3.set_xlabel('number of cities')  # Add an x-label to the axes.
ax3.set_ylabel('error, %')  # Add a y-label to the axes.
ax3.set_title("График зависимости погрешности относительно точного решения от количества городов для ГА.")  # Add a title to the axes.
ax3.legend()
plt.show()


