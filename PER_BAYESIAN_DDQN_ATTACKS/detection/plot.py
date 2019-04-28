import matplotlib.pyplot as plt
import numpy as np
import argparse


import seaborn as sns

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
	file_no_attack = open('uncertainty_no_attack/experiment0.txt' ,mode='r')
	file_no_attack_2 = open('uncertainty_no_attack/experiment1.txt' ,mode='r')

	file_attack = open('uncertainty_attack/experiment0.txt' ,mode='r')
	file_attack_2 = open('uncertainty_attack/experiment1.txt' ,mode='r')

	# read file
	numbers_attack = []
	for x in file_attack:
		numbers_attack.append(float(x))
	i = 0
	for x in file_attack_2:
		if len(numbers_attack) > i:
			break
			
		numbers_attack[i] = (float(x)+numbers_attack[i])/2
		i = i + 1

	print(len(numbers_attack))
	# close the file
	file_attack.close()

	numbers_no_attack = []
	for x in file_no_attack:
		numbers_no_attack.append(float(x))

	i = 0
	for x in file_no_attack_2:
		if len(numbers_attack) > i:
			break
		numbers_no_attack[i] = (float(x)+numbers_no_attack[i])/2
		i += 1
	 
	print(len(numbers_no_attack))
	# close the file
	file_no_attack.close()

	#plt.hist(numbers, color = 'blue', edgecolor = 'black',
    #     bins = 100)

	sns.distplot(numbers_attack, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1}, label='attack', bins=int(len(numbers_attack)*0.01))

	sns.distplot(numbers_no_attack, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1}, label='no_attack', bins=int(len(numbers_no_attack)*0.01))
	#plot
	plt.ylabel('Density')
	plt.xlabel('Uncertainty')
	plt.tight_layout()
	plt.show()
	


if __name__ == "__main__":
	args = parse_args()
	main()