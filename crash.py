from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.traffic_lights import TrafficLights
from flow.envs import Env # Import the base environment class

from gym.spaces.box import Box
from skimage import io
from skimage.color import rgb2gray

import numpy as np

import os
import time


ADDITIONAL_ENV_PARAMS = {
	#maximum acceleration of a rl vehicle
	"max_accel" : 3,
	#maximum deceleration of a rl vechicle
	"max_decel" : 3,
}

class OneJuntionCrashEnv(Env):

	def __init__(self, env_params, sumo_params, scenario):
		self.arrived = 0
		self.crashed = False
		#open a gray initial image of the simulation and retrieve its shape
		screenshot = io.imread(os.getcwd() + "/initial_screenshot.png")
		io.imsave(os.getcwd() + "/screenshot.png", screenshot)
		self.img_shape = screenshot.shape

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
		print('----- dentro do compute-----')
		"""
		Returns the reward of getting into a state. The possible rewards are:
			+1 if a vehicle arrive at its destination
			-1 if a colision happens

		If none of the final states (the vehicle arrive at the destination or happens a colision) have being reached, 
		the state returns no reward.
		"""
		if self.traci_connection.simulation.getCollidingVehiclesNumber() > 0:
			self.crashed = True
			return -1
		elif self.traci_connection.simulation.getArrivedNumber() != 0:
			self.arrived += 1
			return +1
		
		return 0




