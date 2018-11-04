import matplotlib.pyplot as plt
import numpy as np

class plot:

	def plot_one_axis(data, x_label, y_label='steps'):
		x = []
		for i in range(len(data)): x.append(i)
		plt.plot(x, data, '-')
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.show()

	def plot_perf_colli(performance, collision):
		x = []
		for i in range(len(performance)): x.append(i)
		plt.plot(x, performance, '-', x, collision, 'o')
		plt.xlabel('steps')
		plt.ylabel('performance/collision')
		plt.show()
