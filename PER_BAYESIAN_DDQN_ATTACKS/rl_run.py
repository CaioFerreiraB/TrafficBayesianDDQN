import logging
import datetime
import numpy as np

#flow libraries
from flow.core.util import emission_to_csv
from flow.core.experiment import SumoExperiment

#Image libraries
from PIL import Image

#skimage
from skimage import img_as_float

#Pytorch libraries
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

#project libraries
from config import Config
from replay_memory import ReplayMemory
from agent import Agent
from save_logs import SaveLogs
from adversary import *
from detection import *

#system libraries
import os
import time
from copy import deepcopy

#Method to resize the image, used on "get_image"
resize = T.Compose([T.ToPILImage(),
					T.Resize(40, interpolation=Image.CUBIC),
					T.ToTensor()])

class Experiment(SumoExperiment):
	"""
	Class responsable to run the entire experiment. 
	"""

	def __init__(self, env, scenario):
		super().__init__(env, scenario)

	#We have to override the run method from flow in order to use pytorch as our reinforcement learning library
	def run_train_eval(self, num_runs, num_steps, run, saveLogs, train,
			rl_actions=None, convert_to_csv=False, load_path=None):
		"""
        Run the given scenario for a set number of runs and steps per run.

        Parameters
        ----------
            num_runs: int
                number of runs the experiment should perform
            num_steps: int
                number of steps to be performs in each run of the experiment
            train: bool
            	Define if it is a trainning or evaluating experiment
            run: int
            	The number of the current experiment
            saveLogs: SaveLogs object
            	The instance of the package used to save the logs of the simulation
            rl_actions: method, optional
                maps states to actions to be performed by the RL agents (if
                there are any)
            convert_to_csv: bool
                Specifies whether to convert the emission file created by sumo
                into a csv file
            load_path: string
            	Path to the model that should be loaded into the neural network
            	Default: None
        Returns
        -------
            info_dict: dict
                contains returns, average speed per step
		if rl_actions is None:
			def rl_actions(*_):
				return None
		"""
		#1. Initialize the information variables
		info_dict = {}
		rets = []
		mean_rets = []
		ret_lists = []
		vels = []
		mean_vels = []
		std_vels = []

		performance = []
		collisions = []

		#2. Set the reinforcement learning parameters
		action_set = self.env.getActionSet()
		print('LOAD PATH 	--	run:', load_path)
		agent = Agent(action_set, train=True, load_path=load_path)
		target_update_counter = 0

		#3. Initialize the variables that decide when to store the best network
		got_it = 0 # How many times the agent reaches the end of the street
		max_ret = -200
		evaluate_counter = Config.EVALUATE_AMMOUNT
		best_net_state_dict = None

		#4. Run the experiment for a set number of simulations(runs)
		train_simul = 0
		total_simul = 0

		while train_simul < num_runs: 
			total_simul += 1

			#1. initialize the environment
			vel = np.zeros(num_steps)
			logging.info("Iter #" + str(total_simul))
			ret = 0
			ret_list = []
			vehicles = self.env.vehicles
			collision_check = 5

			obs = self.get_screen(self.env.reset())
			self.env.reset_params()
			state = np.stack([obs for _ in range(4)], axis=0)
			#state = torch.from_numpy(obs).to(agent.device).unsqueeze(0)

			if  evaluate_counter == Config.EVALUATE_AMMOUNT:
				train_simul += 1

			#2. Perform one simulation
			for j in range(num_steps):
				print('(episode, step) = ', total_simul, ',', j)
				
				#1. Select and perform an action(the method rl_action is responsable to select the action to be taken)
				if evaluate_counter == Config.EVALUATE_AMMOUNT:
					print('------------ EH IGUAL')
					agent.policy_net.train()
					train = True

				action, Q_value = agent.select_action(self.concatenate(state, agent), train)
				#action, Q_value = agent.select_action(state, train)
				if Q_value is not None and train: saveLogs.save_Q_value(Q_value, run)
				obs, reward, done, _ = self.env.step(action_set[action[0]])

				#2. Convert the observation to a pytorch observation
				obs = self.get_screen(obs)
				reward = torch.tensor([reward], device=agent.device)
				#state = torch.from_numpy(obs).to(agent.device).unsqueeze(0)

				#3. Observe new state
				if not (self.env.arrived or self.env.crashed):
					next_state = []
					next_state.append(obs)
					next_state.append(deepcopy(state[0]))
					next_state.append(deepcopy(state[1]))
					next_state.append(deepcopy(state[2]))
					#next_state = deepcopy(state)
				else:
					next_state = None

				#4. Store the transition in memory
				if train and evaluate_counter == Config.EVALUATE_AMMOUNT:
					agent.append_sample(self.concatenate(state, agent), action, self.concatenate(next_state, agent), reward)
					#agent.append_sample(state, action, next_state, reward)

				#5. Move to the next state
				state = next_state

				#6. Flow code
				vel[j] = np.mean(vehicles.get_speed(vehicles.get_ids()))
				ret += reward
				ret_list.append(reward)

				#7. Perform one step of the optimization (on the target network) if in training mode
				if train and evaluate_counter == Config.EVALUATE_AMMOUNT:
					print('-----ENTROU NA OTIMIZACAO')
					agent.optimize_model()
					agent.policy_net.eval()
					train = False
					target_update_counter += 1
				

				#8. update target network
				if target_update_counter % Config.TARGET_UPDATE == 0:
					target_update_counter = 0
					agent.target_net.load_state_dict(agent.policy_net.state_dict())
					print('update target network ok...')


				#9. Decide if the simulation gets to an end
				if done or self.env.arrived or self.env.crashed:
					agent.episode_durations.append(j + 1)
					if self.env.crashed:
						saveLogs.add_crash()
						print('Crash')
						collision_check = 1
					elif self.env.arrived:
						saveLogs.add_arrive()
						print('all vehicles arrived the destination')
					break

			#3. Decide if the current model of the neural network will be stored
			if self.env.arrived:
				got_it += 1
			else:
				got_it = 0

			print('got_it:', got_it)

			if  evaluate_counter == Config.EVALUATE_AMMOUNT:
				print('--------EVALUATE A: ', evaluate_counter)

				#4. Store information from the simulation
				saveLogs.add_simulation_time(time=j)
				performance.append(j)
				collisions.append(1 if self.env.crashed else 0)
				#5. flow code
				rets.append(ret)
				vels.append(vel)
				mean_rets.append(np.mean(ret_list))
				ret_lists.append(ret_list)
				mean_vels.append(np.mean(vel))
				std_vels.append(np.std(vel))
				#6. save rewards
				#if i % Config.SAVE_REWARDS_FREQUENCy == 0:
				saveLogs.save_reward(ret, run, train_simul)
				saveLogs.save_average_reward(ret)
				saveLogs.save_collision(collision_check, run)
				saveLogs.save_time(j, run)
				evaluate_counter = 0
				got_it = 0

			else:
				print('--------EVALUATE B: ', evaluate_counter)
				evaluate_counter += 1


			if evaluate_counter == Config.EVALUATE_AMMOUNT and got_it == Config.EVALUATE_AMMOUNT and ret > max_ret:
				print('------ENTROU PRA SALVAR')
				max_ret = ret
				saveLogs.save_model(agent.policy_net, agent.optimizer, 10101010, train_simul*j)
				best_net_state_dict = agent.policy_net.state_dict()
				print('got:', got_it)


			

			

		#5. Store the logs of the simulation
		#a. save the final model of the neural network
		saveLogs.save_model(agent.policy_net, agent.optimizer, run, train_simul*j)

		#b. store the data statistics of the simulation
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

	def run_train(self, num_runs, num_steps, run, saveLogs, train,
			rl_actions=None, convert_to_csv=False, load_path=None):
		"""
        Run the given scenario for a set number of runs and steps per run.

        Parameters
        ----------
            num_runs: int
                number of runs the experiment should perform
            num_steps: int
                number of steps to be performs in each run of the experiment
            train: bool
            	Define if it is a trainning or evaluating experiment
            run: int
            	The number of the current experiment
            saveLogs: SaveLogs object
            	The instance of the package used to save the logs of the simulation
            rl_actions: method, optional
                maps states to actions to be performed by the RL agents (if
                there are any)
            convert_to_csv: bool
                Specifies whether to convert the emission file created by sumo
                into a csv file
            load_path: string
            	Path to the model that should be loaded into the neural network
            	Default: None
        Returns
        -------
            info_dict: dict
                contains returns, average speed per step
		if rl_actions is None:
			def rl_actions(*_):
				return None
		"""
		#1. Initialize the information variables
		info_dict = {}
		rets = []
		mean_rets = []
		ret_lists = []
		vels = []
		mean_vels = []
		std_vels = []

		performance = []
		collisions = []

		#2. Set the reinforcement learning parameters
		action_set = self.env.getActionSet()
		print('LOAD PATH 	--	run:', load_path)
		agent = Agent(action_set, train=True, load_path=load_path)
		target_update_counter = 0

		#3. Initialize the variables that decide when to store the best network
		got_it = 0 # How many times the agent reaches the end of the street
		max_ret = -200
		evaluate_counter = Config.EVALUATE_AMMOUNT

		#4. Run the experiment for a set number of simulations(runs)
		for i in range(num_runs):
			#1. initialize the environment
			vel = np.zeros(num_steps)
			logging.info("Iter #" + str(i))
			ret = 0
			ret_list = []
			vehicles = self.env.vehicles
			collision_check = 5

			obs = self.get_screen(self.env.reset())
			self.env.reset_params()
			state = np.stack([obs for _ in range(4)], axis=0)

			#2. Perform one simulation
			for j in range(num_steps):
				print('(episode, step) = ', i, ',', j)
				
				#1. Select and perform an action(the method rl_action is responsable to select the action to be taken)
				action, Q_value = agent.select_action(self.concatenate(state, agent), train)
				if Q_value is not None: saveLogs.save_Q_value(Q_value, run)
				obs, reward, done, _ = self.env.step(action_set[action[0]])

				#2. Convert the observation to a pytorch observation
				obs = self.get_screen(obs)
				reward = torch.tensor([reward], device=agent.device)

				#3. Observe new state
				if not (self.env.arrived or self.env.crashed):
					next_state = []
					next_state.append(obs)
					next_state.append(deepcopy(state[0]))
					next_state.append(deepcopy(state[1]))
					next_state.append(deepcopy(state[2]))
				else:
					next_state = None

				#4. Store the transition in memory
				agent.append_sample(self.concatenate(state, agent), action, self.concatenate(next_state, agent), reward)

				#5. Move to the next state
				state = next_state

				#6. Flow code
				vel[j] = np.mean(vehicles.get_speed(vehicles.get_ids()))
				ret += reward
				ret_list.append(reward)

				#7. Perform one step of the optimization (on the target network) if in training mode
				agent.optimize_model()	

				target_update_counter += 1

				#8. update target network
				if target_update_counter % Config.TARGET_UPDATE == 0:
					target_update_counter = 0
					agent.target_net.load_state_dict(agent.policy_net.state_dict())
					print('update target network ok...')


				#9. Decide if the simulation gets to an end
				if done or self.env.arrived or self.env.crashed:
					agent.episode_durations.append(j + 1)
					if self.env.crashed:
						saveLogs.add_crash()
						print('Crash')
						collision_check = 1
					elif self.env.arrived:
						saveLogs.add_arrive()
						print('all vehicles arrived the destination')
					break


			#4. Store information from the simulation
			saveLogs.add_simulation_time(time=j)
			performance.append(j)
			collisions.append(1 if self.env.crashed else 0)

			#5. flow code
			rets.append(ret)
			vels.append(vel)
			mean_rets.append(np.mean(ret_list))
			ret_lists.append(ret_list)
			mean_vels.append(np.mean(vel))
			std_vels.append(np.std(vel))
			
			#6. save rewards
			#if i % Config.SAVE_REWARDS_FREQUENCy == 0:
			saveLogs.save_reward(rets, run, i)
			saveLogs.save_average_reward(ret)
			saveLogs.save_collision(collision_check, run)
			saveLogs.save_time(j, run)

		#5. Store the logs of the simulation
		#a. save the final model of the neural network
		saveLogs.save_model(agent.policy_net, agent.optimizer, run, i*j)

		#b. store the data statistics of the simulation
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

	def run_eval(self, num_runs, num_steps, run, saveLogs, train, attack, epsilon,
			rl_actions=None, convert_to_csv=False, load_path=None):
		"""
        Run the given scenario for a set number of runs and steps per run.

        Parameters
        ----------
            num_runs: int
                number of runs the experiment should perform
            num_steps: int
                number of steps to be performs in each run of the experiment
            train: bool
            	Define if it is a trainning or evaluating experiment
            run: int
            	The number of the current experiment
            saveLogs: SaveLogs object
            	The instance of the package used to save the logs of the simulation
            rl_actions: method, optional
                maps states to actions to be performed by the RL agents (if
                there are any)
            convert_to_csv: bool
                Specifies whether to convert the emission file created by sumo
                into a csv file
            load_path: string
            	Path to the model that should be loaded into the neural network
            	Default: None
        Returns
        -------
            info_dict: dict
                contains returns, average speed per step
		if rl_actions is None:
			def rl_actions(*_):
				return None
		"""
		#1. Initialize the information variables
		info_dict = {}
		rets = []
		mean_rets = []
		ret_lists = []
		vels = []
		mean_vels = []
		std_vels = []

		performance = []
		collisions = []

		#2. Set the reinforcement learning parameters
		action_set = self.env.getActionSet()
		print('LOAD PATH 	--	run:', load_path)
		agent = Agent(action_set, train=False, load_path=load_path)
		target_update_counter = 0


		#3. Run the experiment for a set number of simulations(runs)
		for i in range(num_runs):
			#1. initialize the environment
			vel = np.zeros(num_steps)
			logging.info("Iter #" + str(i))
			ret = 0
			ret_list = []
			vehicles = self.env.vehicles
			collision_check = 5

			obs = self.get_screen(self.env.reset())
			self.env.reset_params()
			state = np.stack([obs for _ in range(4)], axis=0)

			#2. Perform one simulation
			for j in range(num_steps):
				print('(episode, step) = ', i, ',', j)

				state_conc = self.concatenate(state, agent)

				#0.Attack the images
				if attack:
					state_conc = fgsm_attack(state_conc, epsilon, agent, i*j, saveLogs)
					#state = fgsm_attack(state, epsilon, agent, i*j, saveLogs)

				is_attack, uncertainty, confidence = check_attack(agent, state_conc)
				saveLogs.save_uncertainty(uncertainty, run)

				print('is_attack: ', is_attack)
				
				#1. Select and perform an action(the method rl_action is responsable to select the action to be taken)
				action, Q_value = agent.select_action(state_conc, train=False)
				#action, Q_value = agent.select_action(state, train=False)
				print('action, Q-value:', action, Q_value)
				if Q_value is not None: saveLogs.save_Q_value(Q_value, run)
				obs, reward, done, _ = self.env.step(action_set[action[0]])

				#2. Convert the observation to a pytorch observation
				obs = self.get_screen(obs)
				#state = torch.from_numpy(obs).to(agent.device).unsqueeze(0)
				#print('state shape:', np.shape(obs))
				reward = torch.tensor([reward], device=agent.device)

				#3. Observe new state
				if not (self.env.arrived or self.env.crashed):
					next_state = []
					next_state.append(obs)
					next_state.append(deepcopy(state[0]))
					next_state.append(deepcopy(state[1]))
					next_state.append(deepcopy(state[2]))
					#next_state = deepcopy(state)
				else:
					next_state = None

				#4. Store the transition in memory
				agent.append_sample(self.concatenate(state, agent), action, self.concatenate(next_state, agent), reward)
				#agent.append_sample(state, action, next_state, reward)

				#5. Move to the next state
				state = next_state

				#6. Flow code
				vel[j] = np.mean(vehicles.get_speed(vehicles.get_ids()))
				ret += reward
				ret_list.append(reward)


				#7. Decide if the simulation gets to an end
				if done or self.env.arrived or self.env.crashed:
					agent.episode_durations.append(j + 1)
					if self.env.crashed:
						saveLogs.add_crash()
						print('Crash')
						collision_check = 1
					elif self.env.arrived:
						saveLogs.add_arrive()
						print('all vehicles arrived the destination')
					break

			

			#3. Store information from the simulation
			saveLogs.add_simulation_time(time=j)
			performance.append(j)
			collisions.append(1 if self.env.crashed else 0)

			#4. flow code
			rets.append(ret)
			vels.append(vel)
			mean_rets.append(np.mean(ret_list))
			ret_lists.append(ret_list)
			mean_vels.append(np.mean(vel))
			std_vels.append(np.std(vel))
			
			#5. save rewards
			#if i % Config.SAVE_REWARDS_FREQUENCy == 0:
			saveLogs.save_reward(ret, run, i)
			saveLogs.save_average_reward(ret)
			saveLogs.save_collision(collision_check, run)
			saveLogs.save_time(j, run)

		#4. Store the logs of the simulation
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

	def get_screen(self, screen_image):
		"""
        Format the image observed of the simulation, resizing and setting pytorch dimensions

        Parameters
        ----------
            screen_image: float ndarray
            	the image retrieved from the simulation
        Returns
        -------
            screen: float ndarray
            	the image resize and transposed
		"""

		screen = screen_image.transpose((2, 1, 0))
		screen = T.ToPILImage()(screen)
		#screen = F.to_grayscale(screen, num_output_channels=1)
		screen = T.Resize(150, interpolation=Image.CUBIC)(screen)
		screen = np.array(screen)
		screen = screen.transpose((2, 0, 1))

		return screen
	
	def concatenate(self, data, agent):
		"""
        Concatenate 4 consecutive frames of the simulation

        Parameters
        ----------
            data: float ndarray
            	The frames that will be concatenated
            agent: Agent object
            	The instance of the agent of the simulation. Used to selec the device
        Returns
        -------
            data_append: float ndarray
            	The result of the concatenation
		"""
		if data is not None:
			data_append = np.append(data[0], data[1], axis=1)
			data_append = np.append(data_append, data[2], axis=1)
			data_append = np.append(data_append, data[3], axis=1)

			data_append = torch.from_numpy(data_append).to(agent.device)
			return data_append.unsqueeze(0)