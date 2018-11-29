class Config(object):
    #Network parameters
    BATCH_SIZE = 64
    GAMMA = 0.99 #Discount factor
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10000

    #Rewards/Regrets for the simulation
    COLLISION_REGRET = -100.0
    ARRIVE_REWARD = 50.0
    TIME_PENALTY = -0.001

    #logs parameters
    SAVE_REWARDS_FREQUENCE = 1
    AVERAGE_REWARD_FREQUENCE = 50
