import Genetic_Algorithm as ga
import generate_matrix
import time

n_cities = 10
xy_range = 100

cities, distance = generate_matrix.generate(n_cities, xy_range)

#calculating
start = time.time()
ga_sol, ga_dist = ga.startMethod(cities, n_cities)
end = time.time()

#ploting
ga.plotCities_out(ga_sol)
ga.plotRoute_out(ga_sol)

print('The way:', ga_sol, '\n', 'Dist:', ga_dist, 'км.', '\n', 'Time:', end - start, 'сек.')



