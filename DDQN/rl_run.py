import logging
import datetime
import numpy as np

from flow.core.util import emission_to_csv
from flow.core.experiment import SumoExperiment

from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from config import Config
from replay_memory import ReplayMemory
from agent import Agent
from save_logs import SaveLogs

import os
import time

resize = T.Compose([T.ToPILImage(),
					T.Resize(40, interpolation=Image.CUBIC),
					T.ToTensor()])
#gray_scale = T.Compose([T.ToPILImage(),
#						F.to_grayscale( num_output_channels=1)])

class Experiment(SumoExperiment):

	def __init__(self, env, scenario):
		super().__init__(env, scenario)

	#We have to override the run method from flow in order to use pytorch as our reinforcement learning library
	def run(self, num_runs, num_steps, train, run, model_logs_path, rewards_logs_path, saveLogs, experiment_label='experiment', rl_actions=None, convert_to_csv=False, load_path=None):
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
		agent = Agent(action_set, train=True, load_path=load_path)
		got_it = 0
		got_it_persist = 0
		best_time = 1000

		for i in range(num_runs):
			vel = np.zeros(num_steps)
			logging.info("Iter #" + str(i))
			ret = 0
			ret_list = []
			vehicles = self.env.vehicles
			collision_check = 0

			#obs = self.get_screen(self.env.reset(), agent)
			obs = self.get_screen(self.env.reset(), agent)
			self.env.reset_params()
			#obs_tensor = torch.from_numpy(obs).to(agent.device)
			#state = torch.stack((obs, obs, obs, obs), dim=0)
			state = np.stack([obs for _ in range(4)], axis=0)
			#print('state.shape depois do stack', state.shape)

			for j in range(num_steps):
				print('(episode, step) = ', i, ',', j)
				
				# Select and perform an action(the method rl_action is responsable to select the action to be taken)
				action, Q_value = agent.select_action(self.concatenate(state, agent), train)
				if Q_value is not None: saveLogs.save_Q_value(Q_value, run)
				obs, reward, done, _ = self.env.step(action_set[action[0]])

				# Convert the observation to a pytorch observation
				obs = self.get_screen(obs, agent)
				reward = torch.tensor([reward], device=agent.device)

				# Observe new state
				if not (self.env.arrived or self.env.crashed):
					next_state = []
					next_state.append(obs)
					next_state.append(state[0].copy())
					next_state.append(state[1].copy())
					next_state.append(state[2].copy())
				else:
					next_state = None

				# Store the transition in memory
				agent.memory.push(self.concatenate(state, agent), action, self.concatenate(next_state, agent), reward)

				#Move to the next state
				state = next_state

				# Flow code
				vel[j] = np.mean(vehicles.get_speed(vehicles.get_ids()))
				ret += reward
				ret_list.append(reward)

				# Perform one step of the optimization (on the target network) if in training mode
				if train: agent.optimize_model()

				if done or self.env.arrived or self.env.crashed:
					agent.episode_durations.append(j + 1)
					#plot_durations()
					if self.env.crashed:
						saveLogs.add_crash()
						print('Crash')
						collision_check = 1
						got_it = 0
					elif self.env.arrived:
						saveLogs.add_arrive()
						got_it += 1
						print('all vehicles arrived the destination')
					break

				

			# update target network
			if i % Config.TARGET_UPDATE == 0:
				agent.target_net.load_state_dict(agent.policy_net.state_dict())
				print('update target network ok...')

			if not self.env.arrived and not self.env.crashed:
					got_it = 0
			print(got_it)


			if got_it > 10 and got_it > got_it_persist and j < best_time:
				got_it_persist = got_it
				best_time = j
				saveLogs.save_model(agent.policy_net, agent.optimizer, 10101010)
				print(got_it)


			saveLogs.add_simulation_time(time=j)


			performance.append(j)
			collisions.append(1 if self.env.crashed else 0)

			#flow code
			rets.append(ret)
			vels.append(vel)
			mean_rets.append(np.mean(ret_list))
			ret_lists.append(ret_list)
			mean_vels.append(np.mean(vel))
			std_vels.append(np.std(vel))
			
			# save rewards
			#if i % Config.SAVE_REWARDS_FREQUENCE == 0:
			saveLogs.save_reward(rets, rewards_logs_path, experiment_label, run, i)
			saveLogs.save_average_reward(ret)
			saveLogs.save_collision(collision_check, run)
			saveLogs.save_time(j, run)

			print("Round {0}, return: {1}".format(i, ret))
			print('------------i:', i)


			print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

		saveLogs.save_model(agent.policy_net, agent.optimizer, run)
		#agent.save(step=run, logs_path=model_logs_path, label=experiment_label)

		info_dict["returns"] = np.array(rets.copy())
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

		#print(agent.policy_net.state_dict())

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
		screen = screen_image.transpose((2, 1, 0))
		screen = T.ToPILImage()(screen)
		#screen = F.to_grayscale(screen, num_output_channels=1)
		screen = T.Resize(150, interpolation=Image.CUBIC)(screen)
		screen = np.array(screen)
		screen = screen.transpose((2, 0, 1))

		return screen
	
	def concatenate(self, data, agent):
		if data is not None:
			data_append = np.append(data[0], data[1], axis=1)
			data_append = np.append(data_append, data[2], axis=1)
			data_append = np.append(data_append, data[3], axis=1)

			data_append = torch.from_numpy(data_append).to(agent.device)
			return data_append.unsqueeze(0)