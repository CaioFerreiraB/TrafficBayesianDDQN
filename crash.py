from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.traffic_lights import TrafficLights
from flow.envs import Env # Import the base environment class

import traceback
import logging
import subprocess
try:
    # Load user config if exists, else load default config
    import flow.core.config as config
except ImportError:
    import flow.config_default as config

import traci
from traci import constants as tc
from traci.exceptions import FatalTraCIError, TraCIException

from gym.spaces.box import Box
from skimage import io
from skimage.color import rgb2gray

import numpy as np

import os
import time


ADDITIONAL_ENV_PARAMS = {
	#maximum acceleration of a rl vehicle
	"max_accel" : 4,
	#maximum deceleration of a rl vechicle
	"max_decel" : -1,
}

RETRIES_ON_ERROR = 10

class OneJuntionCrashEnv(Env):

	def __init__(self, env_params, sumo_params, scenario):
		self.arrived = False
		self.crashed = False
		#open a gray initial image of the simulation and retrieve its shape
		screenshot = io.imread(os.getcwd() + "/initial_screenshot.png")
		io.imsave(os.getcwd() + "/screenshot.png", screenshot)
		self.img_shape = screenshot.shape
		self.rl_ids = scenario.vehicles.get_rl_ids().copy()

		super().__init__(env_params, sumo_params, scenario)

	def reset_params(self):
		self.arrived = 0
		self.crashed = False

	def getActionSet(self):
		return [ADDITIONAL_ENV_PARAMS['max_accel'], ADDITIONAL_ENV_PARAMS['max_decel']]

	@property
	def action_space(self):
		return Box(low = -abs(self.env_params.additional_params["max_decel"]),
				   high = self.env_params.additional_params["max_accel"],
				   shape = (self.vehicles.num_rl_vehicles,),
				   dtype=np.float32)

	@property	
	def observation_space(self):
		#The space observed in this simulation is the aerial view of the lanes with cars, so it's just one image object in each observation
		return Box(
			low = -float("inf"),
			high = float("inf"),
			shape = (self.img_shape),
		)

	# For this simulation, the only think that the agente can do is accelerate or decelerate in orther to avoid a crash
	def _apply_rl_actions(self, rl_actions):
		# the names of all autonomous (RL) vehicles in the network
		rl_ids = self.vehicles.get_rl_ids()

		# use the base environment method to convert actions into accelerations for the rl vehicles
		self.apply_acceleration(rl_ids, rl_actions)


	def get_state(self, **kwargs):
		"""
		Return a numpy ndarray containning an image of the simulation at the current step
		"""
		#1. Get the ID's of the views using the TraCI connection to sumo
		VIDs = self.traci_connection.gui.getIDList() 

		#2. Saving the screenshot for each view (usually its just one view). We just save the screenshot of the current step
		#for i, ID in enumerate(VIDs):
		sc_name = os.getcwd() + "/screenshot.png"
		self.traci_connection.gui.screenshot("View #0", sc_name) #VERIFICAR SE ISSO FUNCIONA

		#3. Create a image object (numpy ndarray)
		#IMPORTANT: the screenshot is from the last step performed, not the one just taken!!!!
		screenshot = io.imread(sc_name)
		#screenshot_grey = rgb2gray(screenshot)

		return screenshot

	def compute_reward(self, state, rl_actions, **kwargs):
		"""
		Returns the reward of getting into a state. The possible rewards are:
			+1 if a rl vehicle arrive at its destination
			-1 if a colision happens

		If none of the final states (the vehicle arrive at the destination or happens a colision) have being reached, 
		the state returns no reward.
		"""

		arrived_ids = self.traci_connection.simulation.getArrivedIDList()

		if len(self.traci_connection.simulation.getCollidingVehiclesIDList()) != 0: time.sleep(3)

		if self.traci_connection.simulation.getCollidingVehiclesNumber() > 0:
			self.crashed = True
			return -1.0
		elif any(ID in self.rl_ids for ID in arrived_ids):
			self.arrived = True
			return +1.0
		
		return -0.01

	def start_sumo(self):
		"""Start a sumo instance.

		Uses the configuration files created by the generator class to
		initialize a sumo instance. Also initializes a traci connection to
		interface with sumo from Python.
		"""
		error = None
		for _ in range(RETRIES_ON_ERROR):
			try:
				# port number the sumo instance will be run on
				if self.sumo_params.port is not None:
					port = self.sumo_params.port
				else:
					# Don't do backoff when testing
					if os.environ.get("TEST_FLAG", 0):
						# backoff to decrease likelihood of race condition
						time_stamp = ''.join(str(time.time()).split('.'))
						# 1.0 for consistency w/ above
						time.sleep(1.0 * int(time_stamp[-6:]) / 1e6)
						port = sumolib.miscutils.getFreeSocketPort()

				sumo_binary = "sumo-gui" if self.sumo_params.render else "sumo"

				# command used to start sumo
				sumo_call = [
					sumo_binary, "-c", self.scenario.cfg,
					"--remote-port",
					str(port), "--step-length",
					str(self.sim_step)
				]

				#add a check for collisions in junctions
				sumo_call.append("--collision.check-junctions")
				sumo_call.append("true")

				#take the cars out of the simulation when a collision happens
				sumo_call.append("--collision.action")
				sumo_call.append("remove")

				# add step logs (if requested)
				if self.sumo_params.no_step_log:
					sumo_call.append("--no-step-log")

				# add the lateral resolution of the sublanes (if requested)
				if self.sumo_params.lateral_resolution is not None:
					sumo_call.append("--lateral-resolution")
					sumo_call.append(str(self.sumo_params.lateral_resolution))

				# add the emission path to the sumo command (if requested)
				if self.sumo_params.emission_path is not None:
					ensure_dir(self.sumo_params.emission_path)
					emission_out = \
						self.sumo_params.emission_path + \
						"{0}-emission.xml".format(self.scenario.name)
					sumo_call.append("--emission-output")
					sumo_call.append(emission_out)
				else:
					emission_out = None

				if self.sumo_params.overtake_right:
					sumo_call.append("--lanechange.overtake-right")
					sumo_call.append("true")

				if self.sumo_params.ballistic:
					sumo_call.append("--step-method.ballistic")
					sumo_call.append("true")

				# specify a simulation seed (if requested)
				if self.sumo_params.seed is not None:
					sumo_call.append("--seed")
					sumo_call.append(str(self.sumo_params.seed))

				if not self.sumo_params.print_warnings:
					sumo_call.append("--no-warnings")
					sumo_call.append("true")

				# set the time it takes for a gridlock teleport to occur
				sumo_call.append("--time-to-teleport")
				sumo_call.append(str(int(self.sumo_params.teleport_time)))

				logging.info(" Starting SUMO on port " + str(port))
				logging.debug(" Cfg file: " + str(self.scenario.cfg))
				logging.debug(" Emission file: " + str(emission_out))
				logging.debug(" Step length: " + str(self.sim_step))

				# Opening the I/O thread to SUMO
				self.sumo_proc = subprocess.Popen(
					sumo_call, preexec_fn=os.setsid)

				# wait a small period of time for the subprocess to activate
				# before trying to connect with traci
				if os.environ.get("TEST_FLAG", 0):
					time.sleep(0.1)
				else:
					time.sleep(config.SUMO_SLEEP)

				self.traci_connection = traci.connect(port, numRetries=100)

				self.traci_connection.simulationStep()
				return
			except Exception as e:
				print("Error during start: {}".format(traceback.format_exc()))
				error = e
				self.teardown_sumo()
		raise error




