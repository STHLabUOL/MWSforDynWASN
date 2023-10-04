'''
Evaluate simulation results (RMSE, AMSC, SSNR, ...)
'''

import os
import numpy as np
import pickle
from datetime import datetime
from multiprocessing import Process, Queue

from lazy_dataset.database import JsonDatabase
from paderbox.io import load_audio

from modules.topology_tools import TopologyManager
from modules.eval_utils import get_rto, evaluate_simulation_results

'''
INIT
'''

FLAG_CUSTOM_SIMULATION = False

if FLAG_CUSTOM_SIMULATION:
    # for 50 topologies saved in results/2023_03_24/topologies.pkl used in [1], see GitHub.
    RESULTS_DATA_ROOT = 'results/2023_03_24/'
else:
    # Evaluate WASN signals generated by  simulation_2_simulate_WASNs.py
    # for topologies from topologies_my.pkl
    RESULTS_DATA_ROOT = 'results/'

EVAL_BEFORE_SEGMENT = True # flag for signal segment beeing evaluated
WASN_MODIFICATION = 'join' # 'join', 'unlink', 'leave', 'leave_root'. Please execute for all four possible WASN_MODIFICATIONs
N_PROCS_MAX = 5 # max. number of parallel processes scales memory requirements

SIM_DATA_DIR = RESULTS_DATA_ROOT+'simulation/'+WASN_MODIFICATION+'/'
EVAL_TARGET_DIR = RESULTS_DATA_ROOT+'evaluation/'+('before/' if EVAL_BEFORE_SEGMENT else 'after/'+WASN_MODIFICATION+'/')

if not os.path.isdir(SIM_DATA_DIR):
    raise Exception('Simulation data directory not found.')
if os.path.isdir(EVAL_TARGET_DIR):
    raise Exception('Target directory already exists. Please remove or choose a different directory name.')
else:
    os.makedirs(EVAL_TARGET_DIR)


# Import parameters from simulation metadata
with open(SIM_DATA_DIR+'sim_metadata.pkl', 'rb') as f:
    simdata = pickle.load(f)
sim_type = simdata['sim_type']
sig_len_sec = simdata['sig_len_sec']
fs_Hz = simdata['fs_Hz']
frame_len = simdata['frame_len']
DATA_ROOT = simdata['paths']['data_root']
testbed_json = simdata['paths']['testbed_json']
pos_json = simdata['paths']['pos_json']
ex_id = simdata['example_id']
n_nodes_all = simdata['n_nodes_all']
n_frames = int((sig_len_sec*fs_Hz)/frame_len) 
example_db = JsonDatabase(testbed_json)
examples = example_db.get_dataset('examples')
node_sro = lambda nid: examples[ex_id]['nodes'][nid]['sro']
nodes_select_all = ['node_'+str(nid) for nid in list(range(n_nodes_all))] # all signals should be loaded

# Load async- and synchronous signals for all nodes
signals_async = np.zeros((n_frames, frame_len, n_nodes_all))
signals_sync = np.zeros((n_frames, frame_len, n_nodes_all))
for i in range(n_nodes_all):
    signal_sync = load_audio(DATA_ROOT+'audio/example_0_sync/node_'+str(i)+'_mic_0.wav')[0:(n_frames*frame_len)]
    signals_sync[:,:,i] = np.reshape(signal_sync, (n_frames, frame_len))
    signal_async = load_audio(DATA_ROOT+'audio/example_0_async/node_'+str(i)+'_mic_0.wav')[0:(n_frames*frame_len)]
    signals_async[:,:,i] = np.reshape(signal_async, (n_frames, frame_len))


def process_eval(q, pid, sim_result_file):
    '''
    Computes full evaluation for single topology simulation. Meant to be executed in a separate process.
    Input:
        q (Queue): output queue
        pid (any): arbitrary process id to identify results written in output queue
        sim_result_file (string): path of the simulation result to be evaluated

    '''

    # Load simulation results from file
    with open(sim_result_file, 'rb') as f:
        res = pickle.load(f)
    frame_idx_switch = int(np.floor(res['t_switch']*fs_Hz/frame_len))
    node_ids_ever = res['node_ids_ever']
    nodes_select_all = ['node_'+str(nid) for nid in list(range(n_nodes_all))] 
    nodes_select_before = TopologyManager.get_unique_node_list(res['nodes_levels_before'])
    nodes_select_after = TopologyManager.get_unique_node_list(res['nodes_levels_after'])
    SRO_ref = node_sro(nodes_select_before[0])

    # Set selection masks
    signalsSelect_before = [nodes_select_all.index(nid) for nid in nodes_select_before]
    resultsSelect_before = [node_ids_ever.index(int(nid.split('_')[1])) for nid in nodes_select_before]
    signalsSelect_after = [nodes_select_all.index(nid) for nid in nodes_select_after]
    resultsSelect_after = [node_ids_ever.index(int(nid.split('_')[1])) for nid in nodes_select_after]

    # Get output offset (signal latency) of each node
    #res['resamplerDelay'] = 1 # TEMP!
    node_level_positions_before = TopologyManager.get_node_level_positions(res['nodes_levels_before'])
    node_level_positions_after = TopologyManager.get_node_level_positions(res['nodes_levels_after'])
    node_signals_offset_before = {int(nid_str.split('_')[1]): (lid+1)*res['resamplerDelay'] for nid_str, lid in node_level_positions_before.items()}
    node_signals_offset_after = {int(nid_str.split('_')[1]): (lid+1)*res['resamplerDelay'] for nid_str, lid in node_level_positions_after.items()}

    # Separete "before" and "after" signal segments
    signals_synced_before = res['signals_synced'][:frame_idx_switch,:,:]
    n_frames_before = np.shape(signals_synced_before)[0]
    signals_synced_after = res['signals_synced'][frame_idx_switch+1:,:,:]
    n_frames_after = np.shape(signals_synced_after)[0]

    # Align output signals separately for both segments: compensate offset
    for nid, offset in node_signals_offset_before.items():
        idx = res['node_ids_ever'].index(nid)
        signals_synced_before[:(n_frames_before-offset),:,idx] = signals_synced_before[offset:,:,idx]
    for nid, offset in node_signals_offset_after.items():
        idx = res['node_ids_ever'].index(nid)
        signals_synced_after[:(n_frames_after-offset),:,idx] = signals_synced_after[offset:,:,idx]

    # Calculate additional RTO up until time of network-change to consider in evaluation of "after" segment
    rto_add = []
    rto_add_joined = 0
    for node_idx in resultsSelect_after:
        if node_idx not in resultsSelect_before:
            #joining nodes carry inherent RTO due to lack of resampling during "before" period
            # but only relevant for "joined-only" eval, not "after" eval.
            rto_add.append(0) 
            rto_add_joined = (res['SRO_est'][-1, node_idx]+SRO_ref)*frame_idx_switch*1e-6*frame_len
            continue
        rto_add_late_ = SRO_ref*frame_idx_switch*1e-6*frame_len #b.c. additional resampling for SRO_ref starts only at frame_idx_switch
        rto_add.append(get_rto(res['SRO_est'][:frame_idx_switch, node_idx], frame_len) + rto_add_late_) 


    # Prepare results
    eval_res_before = None
    eval_res_after = None
    eval_res_after_joined = None

    if EVAL_BEFORE_SEGMENT:
        # EVALUATE RESULTS: BEFORE 
        eval_res_before = {'rmse_t': [], 'rmse': [], 'ssnr': [], 'ssnr_async': [], 'amsc': [], 'amsc_async': [], 'Tc': []}
        eval_res_before['rmse_t'], eval_res_before['rmse'], eval_res_before['ssnr'], eval_res_before['ssnr_async'], eval_res_before['amsc'], eval_res_before['amsc_async'], eval_res_before['Tc'] = \
        evaluate_simulation_results(
            nodes_select_before, 
            fs_Hz, 
            frame_len, 
            res['SRO_est'][:frame_idx_switch, resultsSelect_before], 
            res['SRO_true'][resultsSelect_before], 
            SRO_ref,
            signals_async[:frame_idx_switch,:,signalsSelect_before], 
            signals_synced_before[:,:,resultsSelect_before],  #already trimmed
            signals_sync[:frame_idx_switch,:,signalsSelect_before],
            verbose = False
        )
    else:
        # EVALUATE RESULTS: AFTER
        eval_res_after = {'rmse_t': [], 'rmse': [], 'ssnr': [], 'ssnr_async': [], 'amsc': [], 'amsc_async': [], 'Tc': []}
        eval_res_after['rmse_t'], eval_res_after['rmse'], eval_res_after['ssnr'], eval_res_after['ssnr_async'], eval_res_after['amsc'], eval_res_after['amsc_async'], eval_res_after['Tc'] =  \
        evaluate_simulation_results(
            nodes_select_after, 
            fs_Hz, 
            frame_len, 
            res['SRO_est'][frame_idx_switch+1:, resultsSelect_after], 
            res['SRO_true'][resultsSelect_after], 
            SRO_ref,
            signals_async[frame_idx_switch+1:,:,signalsSelect_after], 
            signals_synced_after[:,:,resultsSelect_after], #already trimmed
            signals_sync[frame_idx_switch+1:,:,signalsSelect_after],
            rto_add = rto_add,
            eval_idx_range = range(0, int(10*fs_Hz/frame_len)),
            verbose = False,
            SRO_add = res['SRO_true'][resultsSelect_after][0]-res['SRO_est'][frame_idx_switch+1:, resultsSelect_after][0,0], # corrects for small reference shift when root node changes
        )

        # EVALUTE RESULTS: AFTER, NEWLY JOINED NODES ONLY (evaluated on last 10 seconds like "before" segment)
        if sim_type == "join":
            eval_res_after_joined = {'rmse_t': [], 'rmse': [], 'ssnr': [], 'ssnr_async': [], 'amsc': [], 'amsc_async': [], 'Tc': []}
            signalsSelect_after_joined = nodes_select_all.index('node_'+str(res['node_id_changed']))
            resultsSelect_after_joined = node_ids_ever.index(res['node_id_changed'])
            eval_res_after_joined['rmse_t'], eval_res_after_joined['rmse'], eval_res_after_joined['ssnr'], eval_res_after_joined['ssnr_async'], eval_res_after_joined['amsc'], eval_res_after_joined['amsc_async'], eval_res_after_joined['Tc'] =  \
            evaluate_simulation_results(
                'node_'+str(res['node_id_changed']), 
                fs_Hz, 
                frame_len, 
                np.expand_dims(res['SRO_est'][frame_idx_switch+1:, resultsSelect_after_joined], 1), 
                np.expand_dims(res['SRO_true'][resultsSelect_after_joined], 0), 
                SRO_ref,
                np.expand_dims(signals_async[frame_idx_switch+1:,:,signalsSelect_after_joined], 2), 
                np.expand_dims(signals_synced_after[:,:,resultsSelect_after_joined], 2), #already trimmed
                np.expand_dims(signals_sync[frame_idx_switch+1:,:,signalsSelect_after_joined], 2),
                rto_add = [rto_add_joined],
                verbose = False
            )

    # Return results
    q.put({
        'filename': pid,
        'eval_res_before': eval_res_before,
        'eval_res_after': eval_res_after,
        'eval_res_after_joined': eval_res_after_joined,
    })

def collect_results():
    '''
    Wait for and collect results from output pipe.
    Function accesses and manipulates vars of global scope: q, got_results
    '''
    while True:
        # check if finished
        all_fin = True
        for n in got_results.keys():
            if got_results[n] == False: 
                all_fin = False
                break
        if all_fin == True: break
        print('Waiting for result...')
        res = q.get()
        got_results[res['filename']] = True
        print('Got results for ', res['filename'])
        # save
        with open(EVAL_TARGET_DIR+'eval_'+res['filename'], 'wb') as file:
            pickle.dump(res, file)
    print('Got all results.')


q = Queue()
procs = []
got_results = {}
n_procs_started = 0
directory = os.fsencode(SIM_DATA_DIR)
print('Evaluation started: ' + datetime.now().strftime('%Y-%B-%d %H:%M'))
for nn, file in enumerate(os.listdir(directory)):
    filename = os.fsdecode(file)
    if not filename.endswith(".pkl"):
        print('[Warning] Skipping file with unexpected filetype: ', filename)
        continue
    if filename == 'sim_metadata.pkl':
        continue
    print('Evaluating ', filename)
    procs.append(Process(target=process_eval, args=(q, filename, SIM_DATA_DIR+filename)))
    procs[-1].start()
    n_procs_started += 1
    got_results[filename] = False
    # Intermediate result collect
    if n_procs_started % N_PROCS_MAX == 0:
        collect_results()
# collect remaining results if any are pending
collect_results()
print('Evaluation finished: ' + datetime.now().strftime('%Y-%B-%d %H:%M'))


