'''
Draw groups of nodes for which topologies are generated in the simulation.
'''

import os
import numpy as np
import pickle
import random

'''
INIT
'''

DATA_ROOT = 'data_asmp/'
TARGET_FILE = 'simulation_data/topologies/topologies_50_N_4-6_noBottleneck.pkl'
if os.path.isfile(TARGET_FILE):
    raise Exception('Target file already exists.')

fs_Hz = 16e3 # only used to translate between sample-index and frame-index
frame_len = 2**11 # only used to translate between sample-index and frame-index
N_nodes = 13
t_range_switch = [250, 290]
N_topologies = 50
N_range_init = [4, 6]


'''
DRAW GROUPS
'''

topologies = []
for i in range(N_topologies):

    node_ids_availabe = list(range(N_nodes))
    N_start = random.randint(N_range_init[0], N_range_init[1])
    
    # Sample set of nodes... 
    # ... such that node_9 is always included (to avoid requirements for links through walls)
    node_ids_availabe.pop(9)
    node_ids = [9, *random.sample(node_ids_availabe, N_start-1)]

    # Draw time of switch
    t_switch = random.uniform(t_range_switch[0], t_range_switch[1])
    frame_idx_switch = np.round(t_switch*fs_Hz/frame_len)
    
    # save
    topologies.append({
        'node_ids': node_ids,
        'frame_idx_switch': frame_idx_switch
    })

# Export
with open(TARGET_FILE, 'wb') as f: 
    pickle.dump(topologies, f)