# ABC variables
M_0 = 0  # True mean
S_SQ_0 = 1  # True variation
N = 1000  # Sample size
Sim_per_batch = 10000
Batch_num = 100 # Number of batches
Run_num = 100 # Number of runs
Cut_off = 0.003 # Cut_off quantile for rejection

# Parallel computing settings
agents = 8 
chunk_size = 4