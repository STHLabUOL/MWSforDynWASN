'''
Simulates WASNs for previously drawn node groups.
'''

import numpy as np
import os
from tqdm import tqdm
import pickle
from datetime import datetime

from lazy_dataset.database import JsonDatabase

from modules.audio_reader import AudioReader # customized, previously part of asn_testbed
from modules.sro_estimation import NetworkController
from modules.topology_tools import TopologyManager

'''
INIT
'''

# please execute for all four possible WASN_MODIFICATIONs
WASN_MODIFICATION = 'join' #'join', 'leave', 'unlink', 'leave_root'
DATA_ROOT = 'data/'

# Simulate WASN for 5 topologies saved by simulation_1_draw_topologies.py in topologies_my.pkl
# TOPOLOGIES_FILE = 'results/topologies_my.pkl'
# SIM_TARGET_DATA_ROOT = 'results/simulation/'+WASN_MODIFICATION+'/'

# Simulate WASN for 50 topologies saved in results/2023_03_24/topologies.pkl used for experimantal
# evaluation in publication [1], see GitHub.
TOPOLOGIES_FILE = 'results/2023_03_24/topologies.pkl'
SIM_TARGET_DATA_ROOT = 'results/2023_03_24/simulation/'+WASN_MODIFICATION+'/'

if os.path.isdir(SIM_TARGET_DATA_ROOT):
    raise Exception('Target directory already exists.')
else:
    os.makedirs(SIM_TARGET_DATA_ROOT)


sig_len_sec = 540
fs_Hz = 16e3
frame_len = 2**11
testbed_json = DATA_ROOT+'json/testbed.json'
pos_json = DATA_ROOT+'json/positions.json'
n_frames = int((sig_len_sec*fs_Hz)/frame_len) 
example_db = JsonDatabase(testbed_json)
examples = example_db.get_dataset('examples')
ex_id = 0
node_coord = lambda nid: examples[ex_id]['nodes']["node_"+str(nid)]['position']['coordinates']
node_sro = lambda nid: examples[ex_id]['nodes'][nid]['sro']
n_nodes_all = 13
nodes_select_all = ['node_'+str(nid) for nid in list(range(n_nodes_all))] # all signals should be loaded


print('Simulation started: ' + datetime.now().strftime('%Y-%B-%d %H:%M'))

# Import previously drawn node groups for topologies
with open(TOPOLOGIES_FILE, 'rb') as f:
    topologies = pickle.load(f)

# Save key parameters of this simulation
with open(SIM_TARGET_DATA_ROOT+'sim_metadata.pkl', 'wb') as file:
    metadata = {
        'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S") ,
        'paths': {
            'data_root': DATA_ROOT,
            'testbed_json': testbed_json,
            'pos_json': pos_json,
            'topologies_file': TOPOLOGIES_FILE,
        },
        'example_id': ex_id,
        'n_nodes_all': n_nodes_all,
        'n_topologies': len(topologies),
        'sim_type': WASN_MODIFICATION,
        'sig_len_sec': sig_len_sec,
        'fs_Hz': fs_Hz,
        'frame_len': frame_len,
    }
    pickle.dump(metadata, file)


'''
LOAD AUDIO & GROUND-TRUTH DATA FOR ALL NODES
'''

audio_reader = AudioReader(data_root=DATA_ROOT, block_length=frame_len, block_shift=frame_len, node_ids=nodes_select_all, mic_ids='mic_0')
examples = examples.map(audio_reader)
frame_len = np.shape(examples[ex_id]['audio'][nodes_select_all[0]]['mic_0'])[1]
n_frames_max = min([np.shape(examples[ex_id]['audio'][node_id]['mic_0'])[0] for node_id in nodes_select_all])
n_frames = int((sig_len_sec*fs_Hz)/frame_len) 
if n_frames > n_frames_max:
    n_frames = n_frames_max
    sig_len_prev = sig_len_sec
    sig_len_sec = n_frames*frame_len/fs_Hz
    print('Warning: Audio signals too short for desired simulation length of ', str(sig_len_prev), 's. \nReduced simulation length to ', str(sig_len_sec), 's')
signals = np.stack(tuple(examples[ex_id]['audio'][node]['mic_0'][:n_frames,:] for node in nodes_select_all), axis=2)
print('...Audio signals loaded.')

'''
SIMULATE...
'''


for i, topology in enumerate(topologies): 

    frame_idx_switch = topology['frame_idx_switch']
    t_switch = frame_idx_switch*frame_len/fs_Hz

    # Setup "before" topology
    node_ids = topology['node_ids']
    nodes_levels_before = topology['nodes_levels_before']
    nodes_select_before = TopologyManager.get_unique_node_list(nodes_levels_before)

    if WASN_MODIFICATION == 'leave_root' and topology['modifications']['leave_root']['node_id_changed'] is None:
        print('Skipping ', str(i), ': Root Node 9 cannot be removed (Bottleneck).');
        continue

    # Get modified topology
    node_ids_ever = topology['modifications'][WASN_MODIFICATION]['node_ids_ever']
    node_id_changed = topology['modifications'][WASN_MODIFICATION]['node_id_changed']
    node_link_changed = topology['modifications'][WASN_MODIFICATION]['node_link_changed']
    nodes_levels_after = topology['modifications'][WASN_MODIFICATION]['nodes_levels_after']
    nodes_select_after = TopologyManager.get_unique_node_list(nodes_levels_after)
    n_nodes = max(len(nodes_select_before), len(nodes_select_after))

    # Get ground truth SRO
    SRO_true = np.zeros((n_nodes))
    for j, nid in enumerate(node_ids_ever):
        SRO_true[j] = node_sro('node_'+str(nid)) - node_sro(nodes_select_before[0])

    # Set selection masks
    # One mask for selecting signals among all 13 nodes signals, one mask to select result insertion
    # range relative to only this topologies nodes
    signalsSelect_before = [nodes_select_all.index(nid) for nid in nodes_select_before]
    signalsSelect_after = [nodes_select_all.index(nid) for nid in nodes_select_after]
    resultsSelect_before = [node_ids_ever.index(int(nid.split('_')[1])) for nid in nodes_select_before]
    resultsSelect_after = [node_ids_ever.index(int(nid.split('_')[1])) for nid in nodes_select_after]

    # Prepare results
    signals_synced = np.zeros((n_frames, frame_len, n_nodes))
    SRO_est = np.empty((n_frames, n_nodes))
    SRO_est[:] = np.nan
    dSRO_est = np.empty((n_frames, n_nodes))
    dSRO_est[:] = np.nan

    # Simulate actual WASN 
    Net = NetworkController(nodes_levels_before)
    print('Simulating for Topology ', str(i+1), '/',  len(topologies))
    for frame_idx in tqdm(range(n_frames)):
        if frame_idx == frame_idx_switch:
            Net.restructure(nodes_levels_after)
        signalsSelect = signalsSelect_before if frame_idx < frame_idx_switch else signalsSelect_after
        resultsSelect = resultsSelect_before if frame_idx < frame_idx_switch else resultsSelect_after
        SRO_est_, dSRO_est_, frames_synced = Net.process(signals[frame_idx, :, signalsSelect].T)
        SRO_est[frame_idx, resultsSelect] = SRO_est_
        dSRO_est[frame_idx, resultsSelect] = dSRO_est_
        signals_synced[frame_idx,:, resultsSelect] = frames_synced.T

    # Compile and save results
    res = {
        'sim_type': WASN_MODIFICATION,
        'topology_idx': i,
        'nodes_levels_before': nodes_levels_before,
        'nodes_levels_after': nodes_levels_after,
        'node_ids_ever': node_ids_ever,
        'node_id_changed': node_id_changed,
        'nodes_link_changed': node_link_changed,
        't_switch': t_switch,
        'SRO_true': SRO_true,
        'SRO_est': SRO_est,
        'dSRO_est': dSRO_est,
        'signals_synced': signals_synced,
        'resamplerDelay': Net.resampler_delay
    }
    
    Net.shutdown()

    with open(SIM_TARGET_DATA_ROOT+WASN_MODIFICATION+'_'+str(i+1)+'_'+str(len(topologies))+'.pkl', 'wb') as f:
        pickle.dump(res, f)

print('Simulation finished: ' + datetime.now().strftime('%Y-%B-%d %H:%M'))

