'''
Evaluate simulation results (RMSE, AMSC, SSNR, ...)
'''

import os
import sys
import numpy as np
import pickle
import scipy
from multiprocessing import Process, Queue

from lazy_dataset.database import JsonDatabase
from paderbox.io import load_audio

sys.path.append('modules/')
from online_resampler import OnlineResampler_OA
from topology_tools import TopologyManager

'''
INIT
'''

EVAL_BEFORE_SEGMENT = False
SIM_DATA_ROOT = 'results/2023_03_24/simulation/join/'
EVAL_TARGET_DATA_ROOT = 'results/2023_03_24/evaluation/after/join/'
N_PROCS_MAX = 4 #max. number of parallel processes scales memory requirements.

if not os.path.isdir(SIM_DATA_ROOT):
    raise Exception('Simulation data directory not found.')
if os.path.isdir(EVAL_TARGET_DATA_ROOT):
    raise Exception('Target directory already exists. Please remove or choose a different directory name.')
else:
    os.makedirs(EVAL_TARGET_DATA_ROOT)


# Import parameters from simulation metadata
with open(SIM_DATA_ROOT+'sim_metadata.pkl', 'rb') as f:
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



def gen_offset(sig, offset, fft_size):
    '''
    Generate/compensate STO. Used to compensate RTO.
    '''
    k = np.fft.fftshift(np.arange(-fft_size / 2, fft_size / 2))
    block_len = int(fft_size // 2)
    sig_offset = np.zeros_like(sig)
    block_idx = 0
    len_rest = 0

    integer_offset = np.round(offset)
    rest_offset = integer_offset - offset

    # The STFT uses a Hann window with 50% overlap as analysis window
    win = scipy.signal.windows.hann(block_len, sym=False)

    while True:

        block_start = int(block_idx * block_len / 2 + integer_offset)
        if block_start < 0:
            if block_start + block_len < 0:
                block = np.zeros(block_len)
            else:
                block = np.pad(sig[0:block_start + block_len],
                               (block_len - (block_start + block_len), 0),
                               'constant')
        else:
            if (block_start + block_len) > sig.size:
                block = np.zeros(block_len)
                block[:sig[block_start:].size] = sig[block_start:]
                len_rest = sig[block_start:].size
            else:
                block = sig[block_start:block_start + block_len]

        sig_fft = np.fft.fft(win * block, fft_size)
        sig_fft *= np.exp(-1j * 2 * np.pi * k / fft_size * rest_offset)

        block_start = int(block_idx * block_len / 2)

        if block_start+block_len > sig_offset.size:
            n_pad = block_start + block_len - sig_offset.size
            sig_offset = np.pad(sig_offset, (0, n_pad), 'constant')
            
        sig_offset[block_start:block_start + block_len] += \
            np.real(np.fft.ifft(sig_fft))[:block_len]
        block_end = int(block_idx * block_len / 2 - integer_offset) + block_len

        if block_end > sig.size :
            return sig_offset[:block_start + len_rest]

        block_idx += 1

def get_rto(SRO_est, frame_len, k_max=None):
    '''
    Calculates residual time offset (RTO) in samples given SRO estimate trajectory.
    Input:
        SRO_est (list): SRO estimates over time in ppm
        frame_len (int): Number of signal-samples each estimate corresponds to, i.e. DXCPP-Phat FFT Shift
        k_max (int): Idx up to which the RTO should be computed. The estimate at this index is regarded as the reference SRO.
                    Defaults to list-end index.
    '''
    if k_max is None:
        k_max = len(SRO_est)-1
    return np.sum(SRO_est[k_max]-SRO_est[:k_max])*1e-6*frame_len

def evaluate_simulation_results(
        nodes_select, 
        fs, 
        frame_len, 
        SRO_est, 
        SRO_true,  
        SRO_ref,
        signals_async, 
        signals_synced, 
        signals_sync, 
        verbose = False,
        ref_node_idx = 0, #index of reference node w.r.t. all provided arrays
        eval_idx_range = None, #frame index range of results to evaluate. Default (None) -> last 10 seconds
        rto_add = None, # additional RTO to add to the locally calculated RTO. (i.e. from preceding signal-segments)
        SRO_add = 0, # added to all SRO estimates
    ):

    n_frames, n_nodes = np.shape(SRO_est)

    if eval_idx_range is None:
        eval_last_n_secs = 10 # timespan to consider for amsc, ssnr, ... at the END of the signals
        eval_idx_range = range(n_frames-2-int(eval_last_n_secs*fs/frame_len), n_frames-2) 

    # Correct SRO est. for when current reference nodes SRO estimate deviates from its true relative SRO (because it was elected as reference mid-operation)
    SRO_est[:,1:] = SRO_est[:,1:]+SRO_add

    # RMSE(t)
    mse_t = np.zeros((n_frames, n_nodes))
    rmse_t = np.zeros((n_frames, n_nodes))
    for fr in range(n_frames):
        for n in range(n_nodes):
            if fr == 0:
                mse_t[fr, n] = (SRO_est[fr, n] - SRO_true[n])**2
            else:
                mse_t[fr, n] = 0.9*mse_t[fr-1, n] + 0.1*(SRO_est[fr, n] - SRO_true[n])**2
            rmse_t[fr, n] = np.sqrt(mse_t[fr, n])

    # Estimate residual time offset (RTO)
    rto = []
    for n in range(n_nodes):
        k_max = eval_idx_range[0]
        rto_ = np.sum(SRO_est[k_max, n]-SRO_est[:k_max, n])*1e-6*frame_len if k_max > 0 else 0
        rto_add_ = rto_add[n] if rto_add is not None else 0
        rto.append(rto_ + rto_add_) # every estimate representative of frame_len samples
    if verbose: print("RTO: ", rto)

    # Further resample synced signals to match global reference samplingrate fs=16kHz
    for n in range(n_nodes): 
        Resampler = OnlineResampler_OA()#OnlineResampler_STFT()
        for fr in range(n_frames):
            res_block = Resampler.process(signals_synced[fr,:,n], sro=SRO_ref+SRO_add)
            if fr > Resampler.frameDelay: #compensate for resampler-delay
                signals_synced[fr-Resampler.frameDelay,:,n] = res_block
        del Resampler

    # Obtain RTO compensated, synced signals
    signals_synced_no_rto = np.zeros_like(signals_synced)
    for n in range(n_nodes): 
        s = gen_offset(signals_synced[:,:,n].flatten(), rto[n], 2**13)
        n_frames_shorter = int(np.floor(np.size(s)/frame_len))
        signals_synced_no_rto[:n_frames_shorter,:,n] = np.reshape(s[:(n_frames_shorter*frame_len)], (n_frames_shorter, frame_len)) 

    # Compute settling times (RMSE < 1ppm)
    Tc = []
    for n in range(n_nodes):
        if np.size(np.argwhere(rmse_t[:, n] > 1)) == 0:
            Tc.append(0) #good from the beginning
        elif np.size(np.argwhere(rmse_t[:, n] < 1)) == 0:
            Tc.append(-1) # never converges
        else:
            Tc.append(np.argwhere(rmse_t[:, n] > 1)[-1][0]*frame_len/fs)
    if verbose: print("Tc: ", Tc)


    # RMSE (whole)
    rmse = []
    for n in range(n_nodes):
        rmse.append(np.sqrt(np.mean((SRO_est[eval_idx_range, n] - SRO_true[n])**2)))

    # AMSC (averaged MSC)    
    amsc = []
    amsc_async = []
    for n in range(n_nodes):
        # tmp: debug
        if verbose:
            c = scipy.signal.correlate(signals_sync[eval_idx_range,:,n].flatten(), signals_synced_no_rto[eval_idx_range,:,n].flatten())
            m = np.size(c)
            print('remaining int offset ', np.argmax(c)-m/2)
            #plt.plot(np.linspace(-m/2, m/2, m), c)
            #plt.show()
        _, coh = scipy.signal.coherence(signals_sync[eval_idx_range,:,n].flatten(), signals_synced_no_rto[eval_idx_range,:,n].flatten(), fs, window='hann', nperseg=2**13)
        amsc.append(np.mean(np.abs(coh)**2))
        _, coh_async = scipy.signal.coherence(signals_sync[eval_idx_range,:,n].flatten(), signals_async[eval_idx_range,:,n].flatten(), fs, window='hann', nperseg=2**13)
        amsc_async.append(np.mean(np.abs(coh_async)**2))
    if verbose: print('AMSC: ', amsc); print('AMSC_async: ', amsc_async)

    # Compute global SSNR for all signals
    ssnr = []
    ssnr_async = [] 
    for n in range(n_nodes):
        s = signals_synced_no_rto[eval_idx_range,:,n].flatten()
        s_async = signals_async[eval_idx_range,:,n].flatten()
        # ssnr after sync
        var_sync = np.var(signals_sync[eval_idx_range,:,n].flatten())
        var_diff = np.var(signals_sync[eval_idx_range,:,n].flatten() - s)
        ssnr.append(10*np.log10(var_sync/var_diff))
        # ssnr before sync
        var_diff_async = np.var(signals_sync[eval_idx_range,:,n].flatten() - s_async)
        ssnr_async.append(10*np.log10(var_sync/var_diff_async))
    if verbose: print('SSNR: ', ssnr); print('SSNR_async: ', ssnr_async)


    return rmse_t, rmse, ssnr, ssnr_async, amsc, amsc_async, Tc

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
        with open(EVAL_TARGET_DATA_ROOT+'eval_'+res['filename'], 'wb') as file:
            pickle.dump(res, file)
    print('Got all results.')


q = Queue()
procs = []
got_results = {}
n_procs_started = 0
directory = os.fsencode(SIM_DATA_ROOT)
for nn, file in enumerate(os.listdir(directory)):
    filename = os.fsdecode(file)
    if not filename.endswith(".pkl"):
        print('[Warning] Skipping file with unexpected filetype: ', filename)
        continue
    if filename == 'sim_metadata.pkl':
        continue
    print('Evaluating ', filename)
    procs.append(Process(target=process_eval, args=(q, filename, SIM_DATA_ROOT+filename)))
    procs[-1].start()
    n_procs_started += 1
    got_results[filename] = False
    # Intermediate result collect
    if n_procs_started % N_PROCS_MAX == 0:
        collect_results()
# collect remaining results if any are pending
collect_results()


