import math
import numpy as np
import matplotlib.pyplot as plt

n_cities1 = [6, 7, 8, 9, 10, 11, 12]
dynamic_time1 = [0.223, 0.749, 1.345, 3.274, 10.867, 40.108, 174.417]
ga_time1 = [1264.696, 1650.392, 2240.348, 2857.566, 3503.045, 4250.783, 5023.784]
brut_force_time = [0.827, 4.170, 27.675, 220.889, 2156.303, 22671.339, 258146.393]
ga_error1 = [0.0, 0.0, 0.0, 0.43, 1.7, 1.3, 7.4]

n_cities2 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
dynamic_time2 = [0.118, 0.831, 1.390, 4.412, 11.024, 35.768, 70.765, 200.829, 479.175, 1283.117, 3273.166, 6975.634,
                 16248.152, 54059.737]
ga_time2 = [1242.015, 1677.520, 2205.027, 2931.970, 3620.716, 4360.856, 5199.391, 6117.320, 7060.643, 8144.361,
            9270.473, 10460.979, 11720.879, 13069.174]
ga_error2 = [0.0, 0.0, 0.051, 0.75, 3.4, 2.2, 5.3, 1.2e+01, 1.5e+01, 2.9e+01, 3.5e+01, 6.1e+01, 8.2e+01, 1.1e+02]

'''
    График зависимости количества городов от времени(3 метода)
                                                                '''
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)

ax1.semilogy(n_cities1, ga_time1, label='Genetic algorithm', linewidth=3)
ax1.semilogy(n_cities1, dynamic_time1, label='Dynamic programming method', linewidth=3)
ax1.semilogy(n_cities1, brut_force_time, label='Brut force', linewidth=3)
ax1.grid()
ax1.set_xlabel('number of cities')  # Add an x-label to the axes.
ax1.set_ylabel('time, мс')  # Add a y-label to the axes.
ax1.set_title("График зависимости количества городов от времени для трех методов.")  # Add a title to the axes.
ax1.legend()
plt.show()

'''
    График зависимости количества городов от времени(2 метода)
                                                                '''
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)

ax2.semilogy(n_cities2, ga_time2, label='Genetic algorithm', linewidth=3)
ax2.semilogy(n_cities2, dynamic_time2, label='Dynamic programming method', linewidth=3)
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

ax3.semilogy(n_cities2, ga_error2, label='Genetic algorithm', linewidth=3)
ax3.grid()
ax3.set_xlabel('number of cities')  # Add an x-label to the axes.
ax3.set_ylabel('error')  # Add a y-label to the axes.
ax3.set_title("График зависимости погрешности от количества городов для ГА.")  # Add a title to the axes.
ax3.legend()
plt.show()
