import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import os

class plot:



	def plot_one_axis(path, data, y_label, experiment_label, id_label, x_label='run'):
		x = []
		for i in range(len(data)): x.append(i)
		plt.clf()
		plt.plot(x, data, '-')
		plt.xlabel(x_label)
		plt.ylabel(y_label)

		full_label = experiment_label + '-' + y_label + '-' + id_label +  '.png'
		plt.savefig(path+full_label)

		#plt.show()

	def plot_perf_colli(performance, collision):
		x = []
		for i in range(len(performance)): x.append(i)
		plt.plot(x, performance, '-', x, collision, 'o')
		plt.xlabel('steps')
		plt.ylabel('performance/collision')
		plt.show()




