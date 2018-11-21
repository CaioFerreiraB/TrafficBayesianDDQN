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

import os
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-ml', '--model_logs_path', dest='model_logs_path', help='path of the model checkpoint folder',
						default='./logs/model/', type=str)
	parser.add_argument('-rl', '--rewards_logs_path', dest='rewards_logs_path', help='path of the reward checkpoint folder',
						default='./logs/rewards/', type=str)
	parser.add_argument('-v', '--video_path', dest='video_path', help='path of the video folder',
						default='./rl/video', type=str)
	parser.add_argument('-r', '--restore', dest='restore', help='restore checkpoint',
						default=None, type=str)
	parser.add_argument('-la', '--label', dest='label', help='experiment label',
						default='experiment', type=str)
	parser.add_argument('-lp', '--load_path', dest='load_path', help='path to the model to be loaded',
						default=None, type=str)
	parser.add_argument('-t', '--train', dest='train', help='train policy or not',
						default=True, type=bool)
	args = parser.parse_args()
	return args

def main():
	model_logs_path = args.model_logs_path
	rewards_logs_path = args.rewards_logs_path
	video_path = args.video_path
	restore = args.restore
	label = args.label
	load_path = args.load_path
	train = True

	experiments = 2
	runs = 5000
	steps_per_run = 250

	save_logs = SaveLogs(label, experiments, runs, steps_per_run)

	save_logs.create_logs_path()
	save_logs.create_experiments_logs_path()

	performance = np.zeros(runs)
	collisions = np.zeros(runs)
	rewards = np.zeros(runs)

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

		edges_distribution = ['bottom', 'right'] #set the vehicles to just star on the bottom and right edges
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
		info = exp.run(runs, steps_per_run, run=i, model_logs_path=model_logs_path,
		 				rewards_logs_path=rewards_logs_path, saveLogs=save_logs, train=train,
		 				experiment_label=label, load_path=load_path)
		performance = performance + info['performance']
		collisions = collisions + info['collisions']
		rewards = rewards + info['returns']

		save_logs.save_graph(label, 'exp' + str(i), performance, collisions, rewards)

	performance = performance/experiments
	collisions = collisions/experiments
	rewards = rewards/experiments
	print('final:')
	print('performance:\n', performance)
	print('collisions:\n', collisions)
	print('rewards:\n', rewards)

	save_logs.save_config_and_statistics()

	save_logs.save_graph(label, 'final', performance, collisions, rewards)
	


if __name__ == "__main__":
	args = parse_args()
	main()