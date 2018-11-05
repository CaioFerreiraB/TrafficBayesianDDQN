import logging
import datetime
import numpy as np

from flow.core.util import emission_to_csv
from flow.core.experiment import SumoExperiment

from PIL import Image

import torch
import torchvision.transforms as T

from config import Config
from replay_memory import ReplayMemory
from agent import Agent

resize = T.Compose([T.ToPILImage(),
					T.Resize(40, interpolation=Image.CUBIC),
					T.ToTensor()])

class Experiment(SumoExperiment):

	def __init__(self, env, scenario):
		super().__init__(env, scenario)

	#We have to override the run method from flow in order to use pytorch as our reinforcement learning library
	def run(self, num_runs, num_steps, rl_actions=None, convert_to_csv=False):
		"""
        Run the given scenario for a set number of runs and steps per run.

        Parameters
        ----------
            num_runs: int
                number of runs the experiment should perform
            num_steps: int
                number of steps to be performs in each run of the experiment
            rl_actions: method, optional
                maps states to actions to be performed by the RL agents (if
                there are any)
            convert_to_csv: bool
                Specifies whether to convert the emission file created by sumo
                into a csv file
        Returns
        -------
            info_dict: dict
                contains returns, average speed per step
		if rl_actions is None:
			def rl_actions(*_):
				return None
		"""

		info_dict = {}
		rets = []
		mean_rets = []
		ret_lists = []
		vels = []
		mean_vels = []
		std_vels = []

		performance = []
		collisions = []

		#Set the reinforcement learning parameters
		action_set = self.env.getActionSet()
		agent = Agent(action_set)

		for i in range(num_runs):
			print('episode', i)
			vel = np.zeros(num_steps)
			logging.info("Iter #" + str(i))
			ret = 0
			ret_list = []
			vehicles = self.env.vehicles

			obs = self.get_screen(self.env.reset(), agent)
			self.env.reset_params()
			state = obs - obs
			current_screen = obs

			for j in range(num_steps):
				print('(episode, step) = ', i, ',', j)
				# Select and perform an action(the method rl_action is responsable to select the action to be taken)
				action = agent.select_action(state)
				print('accel' if action[0] == 0 else 'decel')
				obs, reward, done, _ = self.env.step(action_set[action[0]])

				# Convert the observation to a pytorch observation
				obs = self.get_screen(obs, agent)
				reward = torch.tensor([reward], device=agent.device)
				print('obs e reward ok...')

				# Observe new state
				last_screen = current_screen
				current_screen = obs
				if not (self.env.arrived or self.env.crashed):
					next_state = current_screen - last_screen
				else:
					next_state = None
				print('observe next_state ok...')

				# Store the transition in memory
				agent.memory.push(state, action, next_state, reward)
				print('store transition ok...')

				#Move to the next state
				state = next_state

				# Flow code
				vel[j] = np.mean(vehicles.get_speed(vehicles.get_ids()))
				ret += reward
				ret_list.append(reward)

				# Perform one step of the optimization (on the target network)
				agent.optimize_model()
				if done or self.env.arrived or self.env.crashed:
					agent.episode_durations.append(j + 1)
					#plot_durations()
					if self.env.crashed: print('Crash')
					elif self.env.arrived: 	print('all vehicles arrived the destination')
					break

			print('=======================================================================================')
			# update target network
			if i % Config.TARGET_UPDATE == 0:
				agent.target_net.load_state_dict(agent.policy_net.state_dict())
				print('update target network ok...')

			performance.append(j)
			collisions.append(1 if self.env.crashed else 0)

			#flow code
			rets.append(ret)
			vels.append(vel)
			mean_rets.append(np.mean(ret_list))
			ret_lists.append(ret_list)
			mean_vels.append(np.mean(vel))
			std_vels.append(np.std(vel))
			print("Round {0}, return: {1}".format(i, ret))

			print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

		info_dict["returns"] = rets
		info_dict["velocities"] = vels
		info_dict["mean_returns"] = mean_rets
		info_dict["per_step_returns"] = ret_lists
		info_dict["performance"] = np.array(performance.copy())
		info_dict["collisions"] = np.array(collisions.copy())

		print("Average, std return: {}, {}".format(
			np.mean(rets), np.std(rets)))
		print("Average, std speed: {}, {}".format(
			np.mean(mean_vels), np.std(std_vels)))
		self.env.terminate()

		if convert_to_csv:
			# collect the location of the emission file
			dir_path = self.env.sumo_params.emission_path
			emission_filename = \
				"{0}-emission.xml".format(self.env.scenario.name)
			emission_path = \
				"{0}/{1}".format(dir_path, emission_filename)

			# convert the emission file into a csv
			emission_to_csv(emission_path)

		return info_dict

	def get_screen(self, screen_image, agent):
		screen = screen_image.transpose((2, 0, 1))  # transpose into torch order (CHW)
        
		# Convert to float, rescare, convert to torch tensor
		# (this doesn't require a copy)
		screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
		screen = torch.from_numpy(screen)
		# Resize, and add a batch dimension (BCHW)
		return resize(screen).unsqueeze(0).to(agent.device)


	