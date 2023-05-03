import BnB
import Genetic_Algorithm as ga
import generate_matrix
import time

n_cities = 10
xy_range = 100

cities, distance = generate_matrix.generate(n_cities, xy_range)

start = time.time()
ga_sol, ga_dist = ga.startMethod(cities, n_cities)
end = time.time()
print('The way:', ga_sol, '\n', 'Dist:', ga_dist, '\n','Time:', end - start)



