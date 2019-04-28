import matplotlib.pyplot as plt
import numpy as np
import argparse




def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-la', '--label', dest='label', help='experiment label',
						default='experiment', type=str)
	parser.add_argument('-p', '--path', dest='path', help='file path',
						default=None, type=str)
	
	args = parser.parse_args()
	return args

def main():
	# get arguments
	label = args.label
	path = args.path

	# open file
	file = open(path ,mode='r')

	# read file
	numbers = []
	for x in file:
		numbers.append(float(x))
	 
	# close the file
	file.close()

	print(numbers[0])
	print(type(numbers[0]))

	#plot
	plt.plot(numbers, 'bo')
	plt.ylabel('uncertainty')
	plt.show()



if __name__ == "__main__":
	args = parse_args()
	main()