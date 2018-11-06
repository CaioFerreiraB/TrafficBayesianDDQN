class Config(object):
    BATCH_SIZE = 64
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    #Rewards/Regrets for the simulation
    COLLISION_REGRET = -1.0
    ARRIVE_REWARD = 1.0
    TIME_PENALTI = -0.001
