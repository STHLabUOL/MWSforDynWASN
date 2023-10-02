'''
Draw groups of nodes for which topologies are generated in the simulation.
'''

import os
import numpy as np
import pickle
import random
from datetime import datetime
from tqdm import tqdm
from lazy_dataset.database import JsonDatabase
from modules.topology_tools import TopologyManager

'''
INIT
'''

DATA_ROOT = 'data/'
TARGET_FILE = 'results/topologies_my.pkl'
if os.path.isfile(TARGET_FILE):
    raise Exception('Target file already exists.')

fs_Hz = 16e3 # only used to translate between sample-index and frame-index
frame_len = 2**11 # only used to translate between sample-index and frame-index
N_nodes = 13
t_range_switch = [250, 290]
N_topologies = 5
N_range_init = [4, 6]

testbed_json = DATA_ROOT+'json/testbed.json'
pos_json = DATA_ROOT+'json/positions.json'
example_db = JsonDatabase(testbed_json)
examples = example_db.get_dataset('examples')
ex_id = 0
node_coord = lambda nid: examples[ex_id]['nodes']["node_"+str(nid)]['position']['coordinates']



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

'''
DRAW GROUPS
'''

topologies = []
print('Drawing topologies.')
print('Start: ' + datetime.now().strftime('%Y-%B-%d %H:%M'))
for i in tqdm(range(N_topologies)):


    node_ids_availabe = list(range(N_nodes))
    N_start = random.randint(N_range_init[0], N_range_init[1])
    
    # Sample set of nodes... 
    # ... such that node_9 is always included (to avoid requirements for links through walls)
    node_ids_availabe.pop(9)
    node_ids = [9, *random.sample(node_ids_availabe, N_start-1)]
    TopMng = TopologyManager({nid: node_coord(nid) for nid in node_ids})
    nodes_levels_before = TopMng.nodes_levels

    # Draw time of switch
    t_switch = random.uniform(t_range_switch[0], t_range_switch[1])
    frame_idx_switch = int(np.round(t_switch*fs_Hz/frame_len))
    
    # Draw WASN modifications
    nodes_levels_after = {}
    node_id_changed = {}
    node_ids_ever = {}
    for mod_type in ['join', 'unlink', 'leave', 'leave_root']:
        TopMng = TopologyManager({nid: node_coord(nid) for nid in node_ids}) # re-init
        if mod_type == 'leave_root' and TopMng.get_node_ids('int', ordered=True)[0] == 9:
            node_id_changed[mod_type] = None
            nodes_levels_after[mod_type] = None
            node_ids_ever[mod_type] = None
            continue #node 9 cannot be removed
        node_ids_ever[mod_type], node_id_changed[mod_type], node_link_changed = modify_topology(TopMng, N_nodes, mod_type)    
        nodes_levels_after[mod_type] = TopMng.nodes_levels
        
    # save
    topologies.append({
        'node_ids': node_ids,
        'nodes_levels_before': nodes_levels_before,
        'frame_idx_switch': frame_idx_switch,
        'modifications': {
            'join': {
                'node_link_changed': None,
                'node_id_changed': node_id_changed['join'],
                'node_ids_ever': node_ids_ever['join'],
                'nodes_levels_after': nodes_levels_after['join']
            },
            'unlink': {
                'node_link_changed': node_link_changed,
                'node_id_changed': node_id_changed['unlink'],
                'node_ids_ever': node_ids_ever['unlink'],
                'nodes_levels_after': nodes_levels_after['unlink']
            },
            'leave': {
                'node_link_changed': None,
                'node_id_changed': node_id_changed['leave'],
                'node_ids_ever': node_ids_ever['leave'],
                'nodes_levels_after': nodes_levels_after['leave']
            },
            'leave_root': {
                'node_link_changed': None,
                'node_id_changed': node_id_changed['leave_root'],
                'node_ids_ever': node_ids_ever['leave_root'],
                'nodes_levels_after': nodes_levels_after['leave_root']
            }
        }
    })


# Export
with open(TARGET_FILE, 'wb') as f: 
    pickle.dump(topologies, f)


print('Done: ' + datetime.now().strftime('%Y-%B-%d %H:%M'))