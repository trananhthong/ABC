# ABC variables
M_0 = 0  # True mean
S_SQ_0 = 1  # True variation
N = 1000  # Sample size
Run_num = 100 # Number of runs
Cut_off = 0.003 # Cut_off quantile for rejection
Alpha = 8 # Prior's alpha
Beta = 4 # Prior's beta
Batch_size = 10000
Batch_num = 100

# Parallel computing settings
agents = 1
chunk_size = 2
distance_agents = 1
distance_chunk_size = 2
main_agents = 8
main_chunk_size = 1
