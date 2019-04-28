import numpy as np

from flow.core.vehicles import Vehicles
from flow.controllers.car_following_models import IDMController
from flow.controllers.car_following_models import SumoCarFollowingController
from flow.controllers.routing_controllers import GridRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, SumoCarFollowingParams
from flow.core.experiment import SumoExperiment
from flow.controllers import RLController

from one_junction_scenario import OneJunctionScenario #scenario class
from gen_one_junction import OneJunctionGenerator #generator class
from crash import OneJuntionCrashEnv, ADDITIONAL_ENV_PARAMS #environment
from rl_run import Experiment
from plots import plot
from save_logs import SaveLogs
from config import Config

import os
import argparse
import time

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-la', '--label', dest='label', help='experiment label',
						default='experiment', type=str)
	parser.add_argument('-lp', '--load_path', dest='load_path', help='path to the model to be loaded',
						default=None, type=str)
	parser.add_argument('-t', '--train', dest='train', help='train policy or not', choices=['True', 'False'],
						default='True', type=str)
	parser.add_argument('-m', '--mode', dest='mode', help='simulation mode.', choices=['t', 'e', 'te'],
						default='te', type=str)
	parser.add_argument('-a', '--attack', dest='attack', help='Attack or not the images.', choices=['True', 'False'],
						default='False', type=str)
	args = parser.parse_args()
	return args

def main():
	#0. Set initial variables
	label = args.label
	load_path = args.load_path
	mode = args.mode
	if args.train == 'True':
		train = True
	else:
		train = False

	if args.attack == 'True':
		attack = True
	else:
		attack = False

	experiments = 2
	runs = 3000
	steps_per_run = 250

	#1. Set the logs object, creating the logs paths, if it does not exists yet, and the experiments logs path
	save_logs = SaveLogs(label, experiments, runs, steps_per_run)
	save_logs.create_logs_path()
	save_logs.create_experiments_logs_path()

	#2. Iniciate the variables used to store informations from the simulation
	performance = np.zeros(runs)
	collisions = np.zeros(runs)
	rewards = np.zeros(runs)
	q_values = np.zeros(runs)
	loss = np.zeros(runs)

	#3. Run a set number of experiments
	for i in range (experiments):
		#1. Create a Vehicle object containing the vehicles that will be in the simulation
			# The tau parameter must be lower than the simulation step in order to allow collisions
		sumo_car_following_params = SumoCarFollowingParams(sigma=1, security=False, tau=0.1)
		vehicles = Vehicles()
			# The speed mode parameter controlls how the vehicle controlls the speed. All the options for this parameter
			#can be found in: http://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#speed_mode_.280xb3.29
		vehicles.add(veh_id="idm",
	             acceleration_controller=(IDMController, {'T': 0.1, 's0': 0.1}), 
	             routing_controller=(GridRouter, {}),
	             num_vehicles=4,
	             speed_mode=1100,
	             lane_change_mode='aggressive',
	             sumo_car_following_params=sumo_car_following_params)
		vehicles.add(veh_id="rl",
	             acceleration_controller=(RLController, {}),
	             routing_controller=(GridRouter, {}),
	             num_vehicles=1,
	             speed_mode=0000,
	             lane_change_mode='aggressive',
	             sumo_car_following_params=sumo_car_following_params)

		#2. Initite the parameters for a sumo simulation and the initial configurations of the simulation
		sumo_params = SumoParams(sim_step=0.5, render=True)

		edges_distribution = ['bottom', 'right'] #set the vehicles to just start on the bottom and right edges
		initial_config = InitialConfig(edges_distribution=edges_distribution, spacing='custom')


		#3. Set the environment parameter
		env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

		#4. Set the netfile parameters with the path to the .net.xml network file
		net_params = NetParams(netfile=os.getcwd() + '/sumo/one_junction.net.xml')

		#5. Create instances of the scenario and environment
		scenario = OneJunctionScenario(  # we use the NetFileScenario scenario class for one junction scenario... 
		    name="test_NetFile_scenario",
		    generator_class=OneJunctionGenerator,  # ... as well as the newly netfile generator class
		    vehicles=vehicles,
		    net_params=net_params,
		    initial_config=initial_config
		)

		env = OneJuntionCrashEnv(env_params, sumo_params, scenario)

		#6. create a instance of a sumo experiment
		exp = Experiment(env, scenario)

		#7. Run the sumo simulation for a set number of runs and time steps per run
		if mode == 'te':
			info = exp.run_train_eval(runs, steps_per_run, run=i, saveLogs=save_logs, train=True, load_path=load_path)
		elif mode == 't':
			info = exp.run_train(runs, steps_per_run, run=i, saveLogs=save_logs, train=True, load_path=load_path)
		elif mode == 'e':
			info = exp.run_eval(runs, steps_per_run, run=i, saveLogs=save_logs, attack=attack, epsilon=Config.EPSILON_ATTACK, load_path=load_path)
		
		performance = performance + info['performance']
		collisions = collisions + info['collisions']
		rewards = rewards + info['returns']

		save_logs.save_graph(label, 'exp' + str(i), info['performance'], info['collisions'], info['returns'], info['loss'], info['q_values'] )

	#4. Average the total performance of the experiments
	performance = performance/experiments
	collisions = collisions/experiments
	rewards = rewards/experiments

	#5. Store all the statitics of the simulation
	save_logs.save_config_and_statistics()
	save_logs.save_detection_informations()

	#6. Save the graphs produced
	save_logs.save_graph(label, 'final', performance, collisions, rewards, None, None)
	


if __name__ == "__main__":
	args = parse_args()
	main()