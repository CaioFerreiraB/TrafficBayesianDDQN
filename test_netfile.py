from flow.core.vehicles import Vehicles
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import GridRouter
# from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams

vehicles = Vehicles()
vehicles.add(veh_id="idm",
             acceleration_controller=(IDMController, {}),
             routing_controller=(GridRouter, {}),
             num_vehicles=2)

sumo_params = SumoParams(sim_step=0.1, render=True)

edges_distribution = ['bottom', 'right'] #set the vehicles to just star on the bottom and right edges
initial_config = InitialConfig(edges_distribution=edges_distribution, spacing='custom')



from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)



net_params = NetParams(netfile='/mnt/c/Users/caiof/Desktop/IC-Lancaster/TrafficBayesianDDQN/sumo/one_junction.net.xml')




from flow.core.experiment import SumoExperiment

from one_junction_scenario import OneJunctionScenario
from flow.scenarios.netfile.scenario import NetFileScenario
#from flow.scenarios.netfile.gen import NetFileGenerator
from gen_one_junction import OneJunctionGenerator

scenario = OneJunctionScenario(  # we use the NetFileScenario scenario class... 
    name="test_NetFile_scenario",
    generator_class=OneJunctionGenerator,  # ... as well as the newly netfile generator class
    vehicles=vehicles,
    net_params=net_params,
    initial_config=initial_config
)

# AccelEnv allows us to test any newly generated scenario quickly
env = AccelEnv(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

# run the sumo simulation for a set number of time steps
exp.run(1, 1500)