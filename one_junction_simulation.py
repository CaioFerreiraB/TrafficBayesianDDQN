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



def main():
	experiments = 1
	runs = 300
	steps_per_run = 500

	performance = np.zeros(runs)
	collisions = np.zeros(runs)

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
		net_params = NetParams(netfile='/mnt/c/Users/caiof/Desktop/IC-Lancaster/TrafficBayesianDDQN/sumo/one_junction.net.xml')

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
		info = exp.run(runs, steps_per_run)
		performance = performance + info['performance']
		collisions = collisions + info['collisions']

	performance = performance/experiments
	collisions = collisions/experiments
	print('final:')
	print('performance:\n', performance)
	print('collisions:\n', collisions)

	plot.plot_one_axis(performance, 'performance')
	plot.plot_one_axis(collisions, 'collisions')
	


if __name__ == "__main__":
    main()