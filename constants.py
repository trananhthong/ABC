# ABC variables
M_0 = 0  # True mean
S_SQ_0 = 1  # True variation
N = 1000  # Sample size
Run_num = 100 # Number of runs
Cut_off = 0.03 # Cut_off quantile for rejection
Alpha = 8 # Prior's alpha
Beta = 4 # Prior's beta

# Parallel computing settings
agents = 32
chunk_size = 31250
distance_agents = 32
distance_chunk_size = 31250