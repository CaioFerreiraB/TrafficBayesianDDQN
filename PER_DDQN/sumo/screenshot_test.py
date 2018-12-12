from __future__ import absolute_import
from __future__ import print_function

import matplotlib as plt
from skimage import io

import os
import sys
import optparse
import random


# 0- Iniciating SUMO environnment variable path
if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

def run():
	step = 0
    
	print("getIDList")
	print(traci.gui.getIDList())

	while step < 100:
		print(step)
		traci.simulationStep()
        
		if step == 5:
			print('saving screenshot')
			sc_name = os.getcwd() + "/screenshot" + str(step) + ".png"
			traci.gui.screenshot("View #0", sc_name)
			#screenshot = io.imread(sc_name)
			#plt.show(screenshot)

		step += 1
	traci.close()
	sys.stdout.flush()

	screenshot = io.imread(sc_name)
	plt.pyplot.show(screenshot)


def get_options():
	optParser = optparse.OptionParser()
	optParser.add_option(action="store_true",
                         default=False, help="run the commandline version of sumo")
	options, args = optParser.parse_args()
	return options


# this is the main entry point of this script
if __name__ == "__main__":
	#options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
	#if options.nogui:
	#	sumoBinary = checkBinary('sumo')
	#else:
	sumoBinary = checkBinary('sumo-gui')


    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
	traci.start([sumoBinary, "-c", "one_junction.sumocfg"])
	run()