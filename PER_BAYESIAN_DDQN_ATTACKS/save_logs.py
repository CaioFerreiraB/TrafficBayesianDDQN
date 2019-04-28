from skimage import io
import os

from config import Config
from plots import plot

class SaveLogs:

	def __init__(self, experiment_label, num_experiments, runs_per_experiment, steps_per_run, logs_path='./logs/'):
		#simulation params
		self.num_experiments = num_experiments
		self.runs_per_experiment = runs_per_experiment
		self.steps_per_run = steps_per_run
		self.num_simulations = self.num_experiments*runs_per_experiment
		self.average_reward_list = []

		#statistics
		self.num_collisions_total = 0
		self.num_arrived_total = 0
		self.simulation_time_total = 0

		#attack_detection
		self.attacks = 0
		self.attacks_detection = 0
		self.false_positives = 0
		self.true_positives = 0
		self.false_negatives = 0
		self.true_negatives = 0

		#paths
		self.logs_path = logs_path
		self.experiment_label = experiment_label
		self.experiment_logs_path = self.logs_path + self.experiment_label
		self.rewards_path = self.logs_path + self.experiment_label + '/rewards'
		self.Q_value_path = self.logs_path + self.experiment_label + '/Q-value'
		self.collisions_path = self.logs_path + self.experiment_label + '/collisions'
		self.time_path = self.logs_path + self.experiment_label + '/time'
		self.model_path = self.logs_path + self.experiment_label + '/model'
		self.graphs_path = self.logs_path + self.experiment_label + '/graphs'
		self.screnshot_path = self.logs_path + self.experiment_label + '/screenshot'
		self.uncertainty_path = self.logs_path + self.experiment_label + '/uncertainty'
		self.loss_path = self.logs_path + self.experiment_label + '/loss'

		#checkpoints
		self.last_checkpoint_reward = 0

	def create_logs_path(self):
		"""
		Create the logs path if doesn't exists yet

		Parameters
		----------
			logs_path: str
				The path where the logs must be saved

		Returns
		-------
			None
		"""
		os.makedirs(self.logs_path, exist_ok=True)

	def create_experiments_logs_path(self):
		"""
		Create the logs path for the experiment, including rewards, model and graphs path, if doesn't exists yet

		Parameters
		----------
			logs_path: str
				The path where the logs must be saved
					Default: './logs/'
			experiment_label: str
				The label of the experiment

		Returns
		-------
			None
		"""
		os.makedirs(self.experiment_logs_path, exist_ok=True)
		os.makedirs(self.rewards_path, exist_ok=True)
		os.makedirs(self.model_path, exist_ok=True)
		os.makedirs(self.graphs_path, exist_ok=True)
		os.makedirs(self.Q_value_path, exist_ok=True)
		os.makedirs(self.collisions_path, exist_ok=True)
		os.makedirs(self.time_path, exist_ok=True)
		os.makedirs(self.screnshot_path, exist_ok=True)
		os.makedirs(self.uncertainty_path, exist_ok=True)
		os.makedirs(self.loss_path, exist_ok=True)

	def save_config_and_statistics(self):
		file = open(self.experiment_logs_path + '/Config_&_Statistics', 'a+')

		file.write('Experiment: ' + self.experiment_label + '\n')
		file.write('--------------------------------------------\n\n')

		file.write('## Network parameters\n')
		file.write('	BATCH_SIZE = ' +str(Config.BATCH_SIZE)+'\n')
		file.write('	GAMMA = ' + str(Config.GAMMA) + '\n')
		file.write('	EPS_START = ' + str(Config.EPS_START) + '\n')
		file.write('	EPS_END = ' + str(Config.EPS_END) + '\n')
		file.write('	EPS_DECAY = ' + str(Config.EPS_DECAY) + '\n')
		file.write('	TARGET_UPDATE_FREQ = ' + str(Config.TARGET_UPDATE) + '\n\n')

		file.write('## PER parameters\n')
		file.write('	E = ' +str(Config.E)+'\n')
		file.write('	A = ' +str(Config.E)+'\n')
		file.write('	BETA = ' +str(Config.BETA)+'\n')
		file.write('	BETA_INCREMENT = ' +str(Config.BETA_INCREMENT)+'\n\n')

		file.write('## Attacks Detection parameters\n')
		file.write('	STOCHASTIC_PASSES = ' +str(Config.STOCHASTIC_PASSES)+'\n')
		file.write('	UNCERTAINTY_TRESSHOLD = ' +str(Config.UNCERTAINTY_TRESSHOLD)+'\n')
		file.write('	EPSILON_ATTACK = ' +str(Config.EPSILON_ATTACK)+'\n\n')

		file.write('## Rewards/Regrets' + '\n')
		file.write('	COLLISION_REGRET = ' + str(Config.COLLISION_REGRET) + '\n')
		file.write('	ARRIVE_REWARD = ' + str(Config.ARRIVE_REWARD) + '\n')
		file.write('	TIME_PENALTY = ' + str(Config.TIME_PENALTY) + '\n\n')

		file.write('## Experiment parameters' + '\n')
		file.write('	Number of experiments: ' + str(self.num_experiments) + '\n')
		file.write('	Number of runs per experiment: ' + str(self.runs_per_experiment) + '\n')
		file.write('	Total number of simulations: ' + str(self.num_simulations) + '\n\n')
      	
		file.write('## Statistics' + '\n')
		file.write('	- Collisions' + '\n')
		file.write('		Total number of collisions: ' + str(self.num_collisions_total) + '\n')
		file.write('		Average collisions per experiment: ' + str(self.num_collisions_total/self.num_experiments) + '\n')
		file.write('	- Arrives' + '\n')
		file.write('		Total number of arrives: ' + str(self.num_arrived_total) + '\n')
		file.write('		Average arrives per experiment: ' + str(self.num_arrived_total/self.num_experiments) + '\n')
		file.write('	- Vehicles stopped' + '\n')
		file.write('		Total number of vehicles stopped: ' + str(self.num_simulations-self.num_arrived_total-self.num_collisions_total) + '\n')
		file.write('		Average stopped vehicles per experiment: ' + str((self.num_simulations-self.num_arrived_total-self.num_collisions_total)/self.num_experiments) + '\n')
		file.write('	- Time' + '\n')
		file.write('		Average simulation time: ' + str(self.simulation_time_total/(self.num_experiments*self.runs_per_experiment)) + '\n')

		file.close()

	def save_model(self, policy_net, optimizer, experiment, step):
		path = self.model_path + '/' + self.experiment_label + str(experiment) + '.pth'

		policy_net.save(path, step=step, optimizer=optimizer)

	def save_graph(self, experiment_label, label, performance, collisions, rewards, loss, q_values):
		plot.plot_one_axis(self.graphs_path + '/', performance, 'simulation time', experiment_label, label)
		plot.plot_one_axis(self.graphs_path + '/', collisions, 'collisions', experiment_label, label)
		plot.plot_one_axis(self.graphs_path + '/', rewards, 'total reward', experiment_label, label)
		if loss is not None:
			plot.plot_one_axis(self.graphs_path + '/', loss, 'loss', experiment_label, label)
		if q_values is not None:
			plot.plot_one_axis(self.graphs_path + '/', q_values, 'q_value', experiment_label, label)

	def add_crash(self):
		self.num_collisions_total += 1

	def add_arrive(self):
		self.num_arrived_total += 1

	def add_simulation_time(self, time):
		self.simulation_time_total += time

	def save_average_reward(self, reward):
		if len(self.average_reward_list) == Config.AVERAGE_REWARD_FREQUENCE:
			average_reward = sum(self.average_reward_list)/Config.AVERAGE_REWARD_FREQUENCE

			average_reward_log = open(self.rewards_path + '/average_reward.txt', 'a+')
			average_reward_log.write(str(average_reward.data.tolist()[0]) + '\n')

			self.average_reward_list = []
			return

		self.average_reward_list.append(reward)

	def save_reward(self, reward, experiment, current_run):
		reward_log = open(self.rewards_path + '/experiment' + str(experiment) + '.txt', 'a+')
		reward_log.write(str(reward.data.tolist()[0]) + '\n')
		reward_log.close()

	def save_collision(self, collision, experiment):
		reward_log = open(self.collisions_path + '/experiment' + str(experiment) + '.txt', 'a+')
		reward_log.write(str(collision) + '\n')
		reward_log.close()

	def save_Q_value(self, value, experiment):
		reward_log = open(self.Q_value_path + '/experiment' + str(experiment) + '.txt', 'a+')
		reward_log.write(str(value) + '\n')
		reward_log.close()

	def save_time(self, time, experiment):
		reward_log = open(self.time_path + '/experiment' + str(experiment) + '.txt', 'a+')
		reward_log.write(str(time) + '\n')
		reward_log.close()
	
	def save_uncertainty(self, uncertainty, experiment):
		reward_log = open(self.uncertainty_path + '/experiment' + str(experiment) + '.txt', 'a+')
		reward_log.write(str(uncertainty) + '\n')
		reward_log.close()

	def save_uncertainty_attack(self, uncertainty, experiment):
		reward_log = open(self.uncertainty_path + '/attack_experiment' + str(experiment) + '.txt', 'a+')
		reward_log.write(str(uncertainty) + '\n')
		reward_log.close()

	def save_uncertainty_no_attack(self, uncertainty, experiment):
		reward_log = open(self.uncertainty_path + '/no_attack_experiment' + str(experiment) + '.txt', 'a+')
		reward_log.write(str(uncertainty) + '\n')
		reward_log.close()

	def save_loss(self, loss, experiment):
		if loss is not None:
			reward_log = open(self.loss_path + '/experiment' + str(experiment) + '.txt', 'a+')
			reward_log.write(str(loss) + '\n')
			reward_log.close()

	def save_screenshot(self, epsilon, iteration, img):
		"""print('epsilon:', epsilon)
								print('iteration:', iteration)
								print('img.shape:', img.shape)
								print('entrou no screenshot')"""
		#io.imsave(self.screnshot_path + '/' + str(epsilon) + '_' + str(iteration) + '.png', img)
		io.imsave(os.getcwd() + "/attack.png", img)

	def save_detection_informations(self):
		file = open(self.experiment_logs_path + '/detection_statistics', 'a+')

		file.write(str(self.attacks)+'\n')
		file.write(str(self.attacks_detection)+'\n')
		file.write(str(self.false_positives)+'\n')
		file.write(str(self.true_positives)+'\n')
		file.write(str(self.false_negatives)+'\n')
		file.write(str(self.true_negatives)+'\n')

