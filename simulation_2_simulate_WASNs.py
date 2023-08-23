'''
Simulates WASNs for previously drawn node groups.
'''

import numpy as np
import os
import sys
from tqdm import tqdm
import random
import pickle
from datetime import datetime


from asn_testbed.database.database import AsyncWASN
from asn_testbed.database.handler import PositionHandler, TimeHandler

sys.path.append('modules/')
from audio_reader import AudioReader # customized, previously part of asn_testbed
from sro_estimation import NetworkController
from topology_tools import TopologyManager

'''
INIT
'''

SIM_TYPE = 'leave_root' #"join", "leave", "unlink", "leave_root" 
DATA_ROOT = '/home/niklas/asn_testbed_p2/other/2023_ASMP/data_asmp/'
TOPOLOGIES_FILE = '/home/niklas/asn_testbed_p2/other/2023_ASMP/simulation_data/topologies/topologies_50_N_4-6_with_tSwitch_earlier_unique.pkl'
SIM_TARGET_DATA_ROOT = 'results/2023_08_08_testChanges/simulation/'+SIM_TYPE+'/'

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
position_handler = PositionHandler(pos_json)
time_handler = TimeHandler()
example_db = AsyncWASN(testbed_json, position_handler)
examples = example_db.get_scenario('examples')
ex_id = 0
node_coord = lambda nid: position_handler.get_node_pos(examples[ex_id]['nodes']["node_"+str(nid)]['pos_id'])['coordinates']
node_sro = lambda nid: examples[ex_id]['nodes'][nid]['sro']
n_nodes_all = 13
nodes_select_all = ['node_'+str(nid) for nid in list(range(n_nodes_all))] # all signals should be loaded


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
        'sim_type': SIM_TYPE,
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
print('...audio signals loaded.')

'''
SIMULATE...
'''

def modify_topology(TopMng, n_nodes_all, scenario):
    '''
    Applies random topology modofications via TopMng based on given parameters.
    Input:
        TopMng (obj): Reference to Topology Manager
        n_nodes_all (int): number of all available nodes
        scenario (string): which scenario to simulate via the topology change
                           options: "join", "leave", "unlink", "leave_root"
    Return:
        node_ids_ever (list): list of all involved nodes, before or after modification
        node_id_changed (int): which node changed (limited to 1)
        node_link_changed (?): which node-link changed (limited to 1)
    '''

    node_ids = TopMng.get_node_ids()

    if scenario == 'join':
        node_ids_availabe_rest = [nid for nid in range(n_nodes_all) if nid not in node_ids]
        node_id_add = random.sample(node_ids_availabe_rest, 1)[0]
        node_ids_ever = [*node_ids, node_id_add] # all nodes apparing before or after change
        node_id_changed = node_id_add
        node_link_changed = None
        TopMng.add_nodes({node_id_add: node_coord(node_id_add)})

    elif scenario == 'leave' or scenario == 'leave_root':
        node_id_remove = random.sample([nid for nid in node_ids[1:]], 1)[0]
        if scenario == 'leave_root':
            node_id_remove = int(TopMng.nodes_levels[0][0][0].split('_')[1])
        node_ids_ever = node_ids
        node_id_changed = node_id_remove
        node_link_changed = None
        TopMng.remove_nodes([node_id_remove])

    elif scenario == 'unlink':
        node_ids_ever = node_ids
        node_ids_cut = node_ids.copy()
        node_ids_cut.pop(node_ids_cut.index(9)) # unlinking from node 9 forbidden (bottleneck)
        node_link_disable_1 = random.sample(node_ids_cut, 1)[0]
        node_link_disable_2 = random.sample([nid for nid in node_ids_cut if nid != node_link_disable_1], 1)[0]
        node_link_disable = [node_link_disable_1, node_link_disable_2]
        node_id_changed = None
        node_link_changed = node_link_disable
        TopMng.set_node_links([node_link_disable], False)

    return node_ids_ever, node_id_changed, node_link_changed


for i, topology in enumerate(topologies): 


    frame_idx_switch = topology['frame_idx_switch']
    t_switch = frame_idx_switch*frame_len/fs_Hz

    # Setup "before" topology
    node_ids = topology['node_ids']
    TopMng = TopologyManager({nid: node_coord(nid) for nid in node_ids})
    nodes_levels_before = TopMng.nodes_levels
    nodes_select_before = TopologyManager.get_unique_node_list(nodes_levels_before)

    if SIM_TYPE == 'leave_root' and TopMng.get_node_ids('int', ordered=True)[0] == 9:
        print('Skipping ', str(i), ': Root Node 9 cannot be removed (Bottleneck).');
        continue

    # Get modified topology
    node_ids_ever, node_id_changed, node_link_changed = modify_topology(TopMng, n_nodes_all, SIM_TYPE)    
    n_nodes = len(node_ids_ever)
    nodes_levels_after = TopMng.nodes_levels
    nodes_select_after = TopologyManager.get_unique_node_list(nodes_levels_after)

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
        'sim_type': SIM_TYPE,
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

    with open(SIM_TARGET_DATA_ROOT+SIM_TYPE+'_'+str(i+1)+'_'+str(len(topologies))+'.pkl', 'wb') as f:
        pickle.dump(res, f)



