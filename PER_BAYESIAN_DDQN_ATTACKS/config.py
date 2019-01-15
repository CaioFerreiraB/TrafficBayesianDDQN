class Config(object):
    #Network parameters
    MEMORY_SIZE = 1000
    BATCH_SIZE = 64
    GAMMA = 0.99 #Discount factor
    LEARNING_RATE = 0.0025
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10000

    #Prioritized Experience Replay 
    E = 0.01
    A = 0.6
    BETA = 0.3
    BETA_INCREMENT = 0.00001

    #Rewards/Regrets for the simulation
    COLLISION_REGRET = -100.0
    ARRIVE_REWARD = 50.0
    TIME_PENALTY = -0.001

    #logs parameters
    SAVE_REWARDS_FREQUENCY = 1
    AVERAGE_REWARD_FREQUENCE = 50
    EVALUATE_AMMOUNT = 2

    #Attack detection
    STOCHASTIC_PASSES = 100
    UNCERTAINTY_TRESSHOLD = 0.1