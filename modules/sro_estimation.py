'''
Implementation such classes as Node, NetworkController, CL_DXCPPhaT and DXCPPhaT.
'''

import math
from multiprocessing import Process, Queue
import numpy as np
from scipy import signal
from online_resampler import OnlineResampler_STFT, OnlineResampler_OA
from delay_buffer import DelayBuffer
from topology_tools import TopologyManager

class Node():

    '''
    Asynchronous node. Includes SRO Estimator (CL-DXPP) and InputBuffer (Microphone Delay Buffer) to
    delay own microphone signal as required.
    '''
    def __init__(self, id, delay, resamplerType):
        '''
        required delay in frames governed by position in topology (level)
        '''
        self.id = id # string id. format: 'node_x'
        self.SROEstimator = CL_DXCPPhaT(resamplerType=resamplerType, start_delay=delay)
        self.frame_len = self.SROEstimator.DXCPPhaT.FrameSize_input
        self.InputBuffer = DelayBuffer((self.frame_len, delay+1))
        self.frozen = False
        self.freeze_counter = 0 # frames until unfrozen
        self.ell = 0

    def process(self, frame_mic, frame_ref, acs=1):
        '''
        Continue SRO estimation and synchronisation given frame pair of mic- and ref-signal
        '''       
        self.InputBuffer.write(frame_mic.flatten())        
        dSRO_est_, SRO_est_, resamplerShift, frame_mic_synced = \
            self.SROEstimator.process(np.stack((frame_ref, self.InputBuffer.read()), axis=1), acs=acs, resample_only=self.frozen)
        
        # counters
        self.ell += 1
        if self.freeze_counter > 0:
            self.freeze_counter -= 1
            if self.freeze_counter == 0:
                self.set_freeze(False)

        return dSRO_est_, SRO_est_, resamplerShift, frame_mic_synced

    def resize_buffer(self, delay):
        '''
        Resize input buffer; neccessary if position in topology (level) changed.
        '''
        old_InputBuffer = self.InputBuffer
        self.InputBuffer = DelayBuffer((self.frame_len, delay+1))
        # transfer old contents  (some frames will be lost if buffer size decreases, nullframes emerge when size increases)
        for _ in range(old_InputBuffer.length):
            self.InputBuffer.write(old_InputBuffer.read()) 
        del old_InputBuffer

    def set_freeze(self, state=True, n_frames=None):
        '''
        freeze/unfreeze node.
        Frozen nodes pause the SRO est. process under continued resampling
        '''
        self.frozen = state
        if state == True and n_frames is not None:
            self.freeze_counter = n_frames
        if state == False and self.freeze_counter > 0:
            self.freeze_counter = 0


class NetworkController():
    '''
    Manages and simulates a network of nodes for distributed SRO est. and sync.
    Computations are carried out in parallel where possible. 
    Instead of holding a collection of node objects itself, this controller holds a collection
    of parallel processes, each acting as a proxy to one node instance.
    '''

    def __init__(self, nodes_levels=[], frame_len=2**11, resamplerType='stft'):
        self.nodes_levels = nodes_levels
        self.nodes = TopologyManager.get_unique_node_list(nodes_levels) 
        self.n_nodes = len(self.nodes)
        self.n_async_nodes = self.n_nodes-1
        self.resamplerType = resamplerType
        if resamplerType == 'stft':
            self.resampler_delay = OnlineResampler_STFT.defaultParams['fftSizeBlocks']-1+OnlineResampler_STFT.defaultParams['bufferReach'][1]
        elif resamplerType == 'oa':
            self.resampler_delay = OnlineResampler_OA.defaultParams['fftSizeBlocks']-1+OnlineResampler_OA.defaultParams['bufferReach'][1]
        self.node_delays = self._node_delays(nodes_levels, self.nodes)
        self.frame_len = frame_len
        self.frames_freeze = int(90*(16e3/self.frame_len))# how many frames the sro est. process is frozen for, for nodes whose reference changed
        self.ell = 0
        self.inQ = [Queue() for _ in range(self.n_nodes)]
        self.outQ = Queue()
        self.NodeProcs = [Process(target=self.node_process, args=(node_id, self.inQ[i], self.outQ, self.node_delays[i], resamplerType)) for i, node_id in enumerate(self.nodes)]     
        for p in self.NodeProcs:
            p.start()
        # Freeze global root node (global root node should always be frozen)
        self._node_send_task(self.nodes[0], {
            'type': 'freeze',
            'state': True,
        })        

    def shutdown(self):
        '''
        terminates all processes and queues. call this method after simulation end
        '''
        for q in self.inQ:
            q.close()
            q.join_thread()
        self.outQ.close()
        self.outQ.join_thread()
        for p in self.NodeProcs:
            p.terminate()
     
    def _node_send_task(self, node_id, task):
        '''
        Relays task dict to specific node
        '''
        idx = self.nodes.index(node_id)
        self.inQ[idx].put(task)

    def node_process(self, node_id, inQ, outQ, mic_delay, resamplerType):
        '''
        To be spawned in parallel. Manages Node object and handles all associated computations
        Inputs are recieved via the process-specific input Queue. Outputs are written in the shared output Queue.
        '''
        NodeInst = Node(id=node_id, delay=mic_delay, resamplerType=resamplerType)
        # Handle tasks...
        while True:
            req = inQ.get()
            reqType = req['type']
            if reqType == 'process':
                frame_mic = req['frame_mic']
                frame_ref = req['frame_ref'] if 'frame_ref' in req else np.zeros_like(frame_mic)#None
                acs = req['acs'] if 'acs' in req else None
                dSRO_est_, SRO_est_, _, synced_block = NodeInst.process(frame_mic, frame_ref, acs)
                outQ.put({
                    'node_id': NodeInst.id,
                    'dSRO_est': dSRO_est_,
                    'SRO_est': SRO_est_,
                    'synced_block': synced_block
                })
            elif reqType == 'resize_buffer':
                NodeInst.resize_buffer(req['delay'])
            elif reqType == 'freeze':
                n_frames = req['n_frames'] if 'n_frames' in req else None
                NodeInst.set_freeze(req['state'], n_frames)
            elif reqType == 'getResamplerDelay':
                outQ.put({
                    'node_id': NodeInst.id,
                    'resamplerDelay': NodeInst.Estimator.Resampler.frameDelay
                }) 
            else:
                raise Exception('Unknown node request type!')


    def restructure(self, nodes_levels):
        '''
        Modify topology
        '''
        old_NodeProcs = self.NodeProcs
        old_inQ = self.inQ
        old_nodes = self.nodes
        old_nodes_levels = self.nodes_levels

        self.nodes = TopologyManager.get_unique_node_list(nodes_levels)
        self.nodes_levels = nodes_levels
        self.n_nodes = len(self.nodes)
        self.n_async_nodes = self.n_nodes-1
        self.node_delays = self._node_delays(nodes_levels, self.nodes)
        nodes_new = set(self.nodes).difference(set(old_nodes))
        nodes_removed = set(old_nodes).difference(set(self.nodes))

        # Init or restore Asnyc Node Instances
        self.NodeProcs = [None for _ in range(self.n_nodes)]
        self.inQ =  [None for _ in range(self.n_nodes)]
        for nid, node_id in enumerate(self.nodes):
            node_id = self.nodes[nid]
            if node_id not in nodes_new: # is known
                self.NodeProcs[nid] = old_NodeProcs[old_nodes.index(node_id)]
                self.inQ[nid] = old_inQ[old_nodes.index(node_id)]
                if True:
                    self.inQ[nid].put({ # note: would be better to confirm the action...
                        'type': 'resize_buffer',
                        'delay': self.node_delays[nid]
                    })
            else: # is new
                self.inQ[nid] = Queue()
                self.NodeProcs[nid] = Process(target=self.node_process, args=(node_id, self.inQ[nid], self.outQ, self.node_delays[nid], self.resamplerType)) 
                self.NodeProcs[nid].start()
        if None in self.NodeProcs or None in self.inQ:
            raise Exception('Reinitialization of Async Nodes failed.')

        # Terminate processes of removed nodes
        for node_id in nodes_removed:
            old_NodeProcs[old_nodes.index(node_id)].terminate()
            old_inQ[old_nodes.index(node_id)].close()
            old_inQ[old_nodes.index(node_id)].join_thread()

        # Find nodes whose reference has changed to a newly added node
        roots_old = TopologyManager.get_nodes_roots(old_nodes_levels)
        roots_new = TopologyManager.get_nodes_roots(self.nodes_levels)
        nodes_with_changed_root = []
        for node_id, root_node_id in roots_old.items():
            if node_id in roots_new and roots_new[node_id] != root_node_id and roots_new[node_id] in nodes_new:
                nodes_with_changed_root.append(node_id)

        # Temporarily freeze sro. est. process for nodes with changed references
        for node_id in nodes_with_changed_root:
            node_idx = self.nodes.index(node_id)
            self.inQ[node_idx].put({ # note: would be better to confirm the action...
                'type': 'freeze',
                'state': True,
                'n_frames': self.frames_freeze
            })

        # If root node changed: freeze new root (and unfreeze previous root if it's still around)
        if old_nodes[0] != self.nodes[0]: 
            self._node_send_task(self.nodes[0], {
                'type': 'freeze',
                'state': True,
            })  
            if old_nodes[0] in self.nodes: 
                self._node_send_task(old_nodes[0], {
                    'type': 'freeze',
                    'state': False,
                })   
            

    def process(self, frames, acs=1):

        '''
        Note: Order of frames in "frames" is to correspond to the unique node list associated with "nodes_levels", see
        TopologyManager.get_unique_node_list()
        '''

        root_node = self.nodes[0] # 1st level, 1st branch, 1st node
        SRO_est = np.zeros((self.n_nodes,))
        dSRO_est = np.zeros((self.n_nodes,))
        frames_synced = np.zeros_like(frames)

        synced_blocks_prev_level = {}
        for lid, level in enumerate(self.nodes_levels):
            if lid == 0: 
                # obtain global root frame first
                branch = level[0]
                self.inQ[0].put({
                    'type': 'process',
                    'frame_mic': frames[:, 0]
                })
                res = self.outQ.get()
                if res['node_id'] != self.nodes[0]:
                    raise Exception('Could not fetch root node frame. Output Queue may not have been empty...')
                synced_blocks_prev_level[root_node] = res['synced_block'] #.flatten()?
                frames_synced[:,0] = res['synced_block']
                SRO_est[0] = -res['SRO_est']
                dSRO_est[0] = -res['dSRO_est']

            synced_blocks_level = {}
            got_results = {}
            # assign tasks...
            for branch in level:
                ref_node = branch[0]
                synced_block_ref = synced_blocks_prev_level[ref_node]
                for nid, node in enumerate(branch):
                    if nid == 0: continue
                    n = self.nodes.index(node)
                    got_results[node] = False
                    self.inQ[n].put({
                        'type': 'process',
                        'frame_mic': frames[:, n],
                        'frame_ref': synced_block_ref,
                        'acs': acs
                    })

            # collect results for this level...
            while True:
                res = self.outQ.get()
                node_id = res['node_id']
                n = self.nodes.index(node_id)
                got_results[node_id] = True
                synced_blocks_level[node_id] = res['synced_block']
                SRO_est[n] = -res['SRO_est']
                dSRO_est[n] = -res['dSRO_est']
                frames_synced[:, n] = res['synced_block']
                # Check if finished
                all_fin = True
                for nid in got_results.keys():
                    if got_results[nid] == False: 
                        all_fin = False
                        break
                if all_fin == True: break
            synced_blocks_prev_level = synced_blocks_level

        self.ell += 1
        return SRO_est, dSRO_est, frames_synced


    def _node_delays(self, nodes_levels, nodes_select):
        delay_per_level = self.resampler_delay
        node_delays = [0 for _ in nodes_select] #init
        for lid, level in enumerate(nodes_levels):
            delay = (lid+1)*delay_per_level
            for branch in level:
                for nid, node in enumerate(branch):
                    if nid == 0: continue
                    node_delays[nodes_select.index(node)] = delay
        return node_delays
    



class CL_DXCPPhaT():
    '''
    Proposed controller from "CONTROL ARCHITECTURE OF THE DOUBLE-CROSS-CORRELATION PROCESSOR
    FOR SAMPLING-RATE-OFFSET ESTIMATION IN ACOUSTIC SENSOR NETWORKS", 2021. A. Chinaev, S. Wienand, G. Enzner
    '''
    def __init__(self, dxcppParams={}, resamplerType='stft', start_delay=0):
        '''
        start_delay: number of frames to freeze IMC controllers output (to 0) after initialisation.
        this delay should be proportional to the position of the node within the tree/topology to avoid
        resampling based on unreliable results.
        '''
        dxcppParams = dict(DXCPPhaT.defaultParams, **dxcppParams) #assign defaults for missing params
        self.DXCPPhaT = DXCPPhaT(
            RefSampRate_fs_Hz = dxcppParams['RefSampRate_fs_Hz'],      
            FrameSize_input = dxcppParams['FrameSize_input'],        
            FFTshift_dxcp = dxcppParams['FFTshift_dxcp'],         
            FFTsize_dxcp = dxcppParams['FFTsize_dxcp'],           
            AccumTime_B_sec = dxcppParams['AccumTime_B_sec'],            
            ResetPeriod_sec = dxcppParams['ResetPeriod_sec'],           
            SmoConst_CSDPhaT_alpha = dxcppParams['SmoConst_CSDPhaT_alpha'],   
            SmoConst_CSDPhaT_alpha2 = dxcppParams['SmoConst_CSDPhaT_alpha2'],  
            SmoConst_SSOest_alpha = dxcppParams['SmoConst_SSOest_alpha'],    
            AddContWait_NumFr = dxcppParams['AddContWait_NumFr'],          
            SettlingCSD2avg_NumFr = dxcppParams['SettlingCSD2avg_NumFr'],      
            X_12_abs_min = dxcppParams['X_12_abs_min'],           
            SROmax_abs_ppm = dxcppParams['SROmax_abs_ppm'],     
            p_upsmpFac = dxcppParams['p_upsmpFac'],     
            Flag_DisplayResults = dxcppParams['Flag_DisplayResults'] 
        )

        if resamplerType == 'stft':
            self.Resampler = OnlineResampler_STFT()
        elif resamplerType == 'oa':
            self.Resampler = OnlineResampler_OA()
        else:
            raise Exception('Unknown resampler type!')
        
        self.zjBuffer = DelayBuffer((self.DXCPPhaT.FrameSize_input, 1+self.Resampler.frameDelay)) # Resampler adds delay
        self.dSRO_est = np.zeros(3) # buffered, internal
        self.SRO_est = np.zeros(3) # buffered, internal
        self.dSRO_est_curr = 0 # current, out
        self.dSRO_est_curr_raw = 0 # raw DXCPP output, not affected by ACS
        self.SRO_est_curr = 0 #current, out
        self.SRO_est_op = 0 # in MATLAB "SROppm_init"
        self.STO_est = 0
        # IMC Controller (Variant 'PIT1'/Tf=8)
        self.K_Nom = [0, 0.0251941968627353, -0.0249422548941180]
        self.K_Denom = [1, -1.96825464010938, 0.968254640109407]
        self.maxDelta = 10**6 / (self.DXCPPhaT.AccumTime_B_NumFr*self.DXCPPhaT.FFTshift_dxcp)
        self.L_hold = 100
        self.forwardControl = False
        self.forwardControlStepCount = 0
        self.start_delay = start_delay 
        self.ell = 0


    def process(self, x_12_ell, acs=1, resample_only=False):

        '''
        Resample z_i, delay z_j, get and return DXCPP Results together with synced frame.
        Note: The returned synced z_i frame will be the frame from 2 iterations earlier, 
        resampled based on the previously estimated SRO (not the updated SRO estimate resulting from this call) 
        
        acs: ACS value, 0 or 1, for controlling DXCPP output (dSRO).
        '''

        # Resample z_i based on latest SRO estimate
        z_i = self.Resampler.process(x_12_ell[:, 1].flatten(), -self.SRO_est_curr)
        if resample_only:
            return 0, self.SRO_est_curr, self.Resampler.shift, z_i

        x_12_ell[:, 1] = z_i    
        self.zjBuffer.write(x_12_ell[:, 0].flatten()) # Delay z_j to accomodate delay of z_i caused by resampler
        x_12_ell[:, 0] = self.zjBuffer.read()  

        # Estimate residual SRO (Buffer filled from left)
        res = self.DXCPPhaT.process_data(x_12_ell)
        self.dSRO_est_curr_raw = res['SROppm_est_out']
        self.dSRO_est_curr = res['SROppm_est_out'] if acs == 1 else 0 #force dSRO=0 in bad conditions
        if self.ell <= self.start_delay:
            self.dSRO_est_curr = 0
        self.dSRO_est[1:] = self.dSRO_est[:-1]
        self.dSRO_est[0] = self.dSRO_est_curr
        self.SRO_est[1:] = self.SRO_est[:-1]
        self.SRO_est[0] = np.dot(self.K_Nom, self.dSRO_est) - np.dot(self.K_Denom[1:], self.SRO_est[1:]) 
        self.SRO_est_curr = self.SRO_est[0] + self.SRO_est_op

        self.ell += 1
        
        return self.dSRO_est_curr, self.SRO_est_curr, self.Resampler.shift, z_i



class DXCPPhaT():

    defaultParams = {
        'RefSampRate_fs_Hz': 16000,      # reference sampling rate
        'FrameSize_input': 2048,         # frame size (power of 2) of input data
        'FFTshift_dxcp': 2**11,          # frame shift of DXCP-PhaT (power of 2 & >= FrameSize_input)
        'FFTsize_dxcp': 2**13,           # FFT size of DXCP-PhaT (power of 2 & >= FFTshift_dxcp)
        'AccumTime_B_sec': 5,            # accumulation time in sec (usually 5s as in DXCP)
        'ResetPeriod_sec': 30,           # resetting period of DXCP-PhaT in sec. Default: 30 (>=2*AccumTime_B_sec)
        'SmoConst_CSDPhaT_alpha': .5,    # smoothing constant for GCSD1 averaging (DXCP-PhaT)
        'SmoConst_CSDPhaT_alpha2': .99,#.978,  # smoothing constant for GCSD2 averaging (DXCP-PhaT)
        'SmoConst_SSOest_alpha': .99,    # smoothing constant of SRO-comp. CCF-1 used to estimate d12 (DXCP-PhaT) [.995 for big mic-dist]
        'AddContWait_NumFr': 0,          # additional waiting for container filling (>InvShiftFactor-1)
        'SettlingCSD2avg_NumFr': 4,      # settling time of CSD-2 averaging (SettlingCSD2avg_NumFr < Cont_NumFr-AddContWait_NumFr)
        'X_12_abs_min': 1e-12,           # minimum value of |X1*conj(X2)| to avoid devision by 0 in GCC-PhaT
        'SROmax_abs_ppm': 1000,          # maximum absolute SRO value possible to estimate (-> Lambda)
        'p_upsmpFac': 4,
        'Flag_DisplayResults': 1 
    }

    def __init__(
        self, 
        RefSampRate_fs_Hz = defaultParams['RefSampRate_fs_Hz'],      
        FrameSize_input = defaultParams['FrameSize_input'],        
        FFTshift_dxcp = defaultParams['FFTshift_dxcp'],         
        FFTsize_dxcp = defaultParams['FFTsize_dxcp'],           
        AccumTime_B_sec = defaultParams['AccumTime_B_sec'],            
        ResetPeriod_sec = defaultParams['ResetPeriod_sec'],           
        SmoConst_CSDPhaT_alpha = defaultParams['SmoConst_CSDPhaT_alpha'],   
        SmoConst_CSDPhaT_alpha2 = defaultParams['SmoConst_CSDPhaT_alpha2'],  
        SmoConst_SSOest_alpha = defaultParams['SmoConst_SSOest_alpha'],    
        AddContWait_NumFr = defaultParams['AddContWait_NumFr'],          
        SettlingCSD2avg_NumFr = defaultParams['SettlingCSD2avg_NumFr'],      
        X_12_abs_min = defaultParams['X_12_abs_min'],           
        SROmax_abs_ppm = defaultParams['SROmax_abs_ppm'],     
        p_upsmpFac = defaultParams['p_upsmpFac'],     
        Flag_DisplayResults = defaultParams['Flag_DisplayResults']         
    ):

        # General parameters (config)
        self.RefSampRate_fs_Hz = RefSampRate_fs_Hz      
        self.FrameSize_input = FrameSize_input
        self.FFTshift_dxcp = FFTshift_dxcp        
        self.FFTsize_dxcp = FFTsize_dxcp         
        self.AccumTime_B_sec = AccumTime_B_sec            
        self.ResetPeriod_sec = ResetPeriod_sec           
        self.SmoConst_CSDPhaT_alpha = SmoConst_CSDPhaT_alpha   
        self.SmoConst_CSDPhaT_alpha2 = SmoConst_CSDPhaT_alpha2  
        self.SmoConst_SSOest_alpha = SmoConst_SSOest_alpha    
        self.AddContWait_NumFr = AddContWait_NumFr
        self.SettlingCSD2avg_NumFr = SettlingCSD2avg_NumFr
        self.X_12_abs_min = X_12_abs_min
        self.SROmax_abs_ppm = SROmax_abs_ppm
        self.p_upsmpFac = p_upsmpFac
        self.Flag_DisplayResults = Flag_DisplayResults

        # Implicit parameters
        self.LowFreq_InpSig_fl_Hz = .01 * self.RefSampRate_fs_Hz / 2
        self.UppFreq_InpSig_fu_Hz = .95 * self.RefSampRate_fs_Hz / 2
        self.RateDXCPPhaT_Hz = self.RefSampRate_fs_Hz / self.FFTshift_dxcp
        self.AccumTime_B_NumFr = int(self.AccumTime_B_sec // (1 / self.RateDXCPPhaT_Hz))
        self.B_smpls = self.AccumTime_B_NumFr * self.FFTshift_dxcp
        self.Upsilon = int(self.FFTsize_dxcp / 2 - 1)
        self.Lambda = int(((self.B_smpls * self.SROmax_abs_ppm) // 1e6) + 1)
        self.Cont_NumFr = self.AccumTime_B_NumFr + 1
        self.InvShiftFactor_NumFr = int(self.FFTsize_dxcp / self.FFTshift_dxcp)
        self.ResetPeriod_NumFr = int(self.ResetPeriod_sec // (1 / self.RateDXCPPhaT_Hz))
        self.FFT_Nyq = int(self.FFTsize_dxcp / 2 + 1)
        self.FreqResol = self.RefSampRate_fs_Hz / self.FFTsize_dxcp
        self.LowFreq_InpSig_fl_bin = int(self.LowFreq_InpSig_fl_Hz // self.FreqResol)
        self.UppFreq_InpSig_fu_bin = int(self.UppFreq_InpSig_fu_Hz // self.FreqResol)
        self.NyqDist_fu_bin = self.FFT_Nyq - self.UppFreq_InpSig_fu_bin    

        # STATE: SCALARS
        self.SROppm_est_ell = 0 # current SRO estimate
        self.SSOsmp_est_ell = 0 # current SSO estimate

        # STATE: MULTIDIM
        self.GCSD_PhaT_avg = np.zeros((self.FFTsize_dxcp, 1), dtype=complex)                 # Averaged CSD with Phase Transform        
        self.GCSD_PhaT_avg_Cont = np.zeros((self.FFTsize_dxcp, self.Cont_NumFr), dtype=complex)   # Container with past GCSD_PhaT_avg values        
        self.GCSD2_avg = np.zeros((self.FFTsize_dxcp, 1), dtype=complex)                     # averaged CSD-2        
        self.GCCF1_smShftAvg = np.zeros((2 * self.Upsilon + 1, 1), dtype=float)              # smoothed shifted first CCF      
        self.GCCF2_avg_ell_big = np.zeros(17, dtype=float)                              # time-domain cff2 for debugging  
        self.InputBuffer = np.zeros((self.FFTsize_dxcp, 2), dtype=float)                     # input buffer of DXCP-PhaT-CL

        # STATE: COUNTERS AND FLAGS      
        self.flag_initiated = False         # flag for initialized recursive averaging.  
        self.ell_execDXCPPhaT = 1           # counter for filling of the input buffer before DXCP-PhaT can be executed        
        self.ell = 1                        # counter within signal     

    def process_data(self, x_12_ell, tdoa=0): # process data of the current input frame

        # fill the internal buffer of DXCP-PhaT-CL (from right to left)
        self.InputBuffer[np.arange(0, self.FFTsize_dxcp - self.FrameSize_input), :] = self.InputBuffer[np.arange(self.FrameSize_input, self.FFTsize_dxcp), :]
        self.InputBuffer[np.arange(self.FFTsize_dxcp - self.FrameSize_input, self.FFTsize_dxcp), :] = x_12_ell

        # execute DXCP-PhaT-CL when enough input frames are collected
        if self.ell_execDXCPPhaT == int(self.FFTshift_dxcp / self.FrameSize_input):
            self.ell_execDXCPPhaT = 0
            self._stateupdate(tdoa)

        # update counter for filling of the input buffer
        self.ell_execDXCPPhaT += 1

        # compose output
        OutputDXCPPhaTcl = {}
        OutputDXCPPhaTcl['SROppm_est_out'] = self.SROppm_est_ell
        OutputDXCPPhaTcl['STOsmp_est_out'] = self.SSOsmp_est_ell
        #OutputDXCPPhaTcl['GCCF2_avg'] = self.GCCF2_avg_ell_big
        return OutputDXCPPhaTcl


    def _stateupdate(self, tdoa=0): 

        # (1) Windowing to the current frames acc. to eq. (5) in [1]
        analWin = signal.blackman(self.FFTsize_dxcp, sym=False)
        x_12_win = self.InputBuffer * np.vstack((analWin, analWin)).transpose()
        
        # (2) Estimate generalized (normalized) GCSD with Phase Transform (GCSD-PhaT) via recursive averaging
        X_12 = np.fft.fft(x_12_win, self.FFTsize_dxcp, 0)
        X_12_act = X_12[:, 0] * np.conj(X_12[:, 1])
        X_12_act_abs = abs(X_12_act)
        X_12_act_abs[X_12_act_abs < self.X_12_abs_min] = self.X_12_abs_min  # avoid division by 0
        GCSD_PhaT_act = X_12_act / X_12_act_abs
        if self.flag_initiated == False:
            self.GCSD_PhaT_avg = GCSD_PhaT_act
        else:
            self.GCSD_PhaT_avg = self.SmoConst_CSDPhaT_alpha * self.GCSD_PhaT_avg + (1 - self.SmoConst_CSDPhaT_alpha) * GCSD_PhaT_act
        # (3) Fill the DXCP-container with self.Cont_NumFr number of past GCSD_PhaT_avg
        self.GCSD_PhaT_avg_Cont[:, np.arange(self.Cont_NumFr - 1)] = self.GCSD_PhaT_avg_Cont[:, 1:]
        self.GCSD_PhaT_avg_Cont[:, (self.Cont_NumFr - 1)] = self.GCSD_PhaT_avg
    
        # (4) As soon as DXCP-container is filled with resampled data, calculate the second GCSD based
        # on last and first vectors of DXCP-container and perform time averaging
        if self.ell >= self.Cont_NumFr + (self.InvShiftFactor_NumFr - 1) + self.AddContWait_NumFr:
            # Estimate second GCSD via recursive averaging
            GCSD2_act = self.GCSD_PhaT_avg_Cont[:, -1] * np.conj(self.GCSD_PhaT_avg_Cont[:, 0])
            if self.flag_initiated == False:
                self.GCSD2_avg[:, 0] = GCSD2_act
            else:
                self.GCSD2_avg[:, 0] = self.SmoConst_CSDPhaT_alpha2 * self.GCSD2_avg[:, 0] + (1 - self.SmoConst_CSDPhaT_alpha2) * GCSD2_act
            # remove non-coherent components
            GCSD2_avg_ifft = self.GCSD2_avg
            # set lower frequency bins (w.o. coherent components) to 0
            GCSD2_avg_ifft[np.arange(self.LowFreq_InpSig_fl_bin), 0] = 0
            GCSD2_avg_ifft[np.arange(self.FFTsize_dxcp - self.LowFreq_InpSig_fl_bin + 1, self.FFTsize_dxcp), 0] = 0
            # set upper frequency bins (w.o. coherent components) to 0
            GCSD2_avg_ifft[np.arange(self.FFT_Nyq - self.NyqDist_fu_bin - 1, self.FFT_Nyq + self.NyqDist_fu_bin), 0] = 0
            # Calculate averaged CCF-2 in time domain
            GCCF2_avg_ell_big = np.fft.fftshift(np.real(np.fft.ifft(GCSD2_avg_ifft, n=self.FFTsize_dxcp, axis=0)))
            idx = np.arange(self.FFT_Nyq - self.Lambda - 1, self.FFT_Nyq + self.Lambda)
            GCCF2avg_ell = GCCF2_avg_ell_big[idx, 0]
            # Log in state for debugging
            self.GCCF2_avg_ell_big = GCCF2avg_ell[72:89] # only middle-part
    
  
        # (5) Parabolic interpolation (13) with (14) with maximum search as in [1]
        # and calculation of the remaining current SRO estimate sim. to (15) in [1]
        # As soon as GCSD2_avg is smoothed enough in every reseting section
        if self.ell >= self.Cont_NumFr + (self.InvShiftFactor_NumFr - 1) + self.AddContWait_NumFr + self.SettlingCSD2avg_NumFr:
            # p-fold upsampling
            upsmpWindow = signal.get_window(('kaiser', 5.0), Nx=2*self.Lambda+1, fftbins=False)
            GCCF2avg_ell_upsmp = signal.resample(GCCF2avg_ell, num=(2*self.Lambda+1)*self.p_upsmpFac, window=upsmpWindow)
            lambda_vec_upsmp = np.arange(-self.Lambda, self.Lambda+1, 1/self.p_upsmpFac) # upsampled lambda-scale
            # maximum search
            idx_max = GCCF2avg_ell_upsmp.argmax(0)
            #if (idx_max == 0) or (idx_max == 2 * self.Lambda):
            if (idx_max == 0) or (idx_max == len(lambda_vec_upsmp)-1):
                DelATSest_ell_frac = 0
            else:
                # set supporting points for search of real-valued maximum
                # NOTICE: This needs fixing! Out-of-range error occurs when maximum itself is at the border
                #         limit idx_max +2 and -1 for these edge-cases
                sup_pnts = GCCF2avg_ell_upsmp[np.arange(idx_max - 1, idx_max + 2)]  # supporting points y(x) for x={-1,0,1}
                # calculate fractional of the maximum via x_max=-b/2/a for y(x) = a*x^2 + b*x + c
                DelATSest_ell_frac = (sup_pnts[2, ] - sup_pnts[0, ]) / 2 / ( 2 * sup_pnts[1, ] - sup_pnts[2, ] - sup_pnts[0, ])

            # [old] w/o resmp: DelATSest_ell = lambda_vec_upsmp[idx_max] - self.Lambda + DelATSest_ell_frac  # resulting real-valued x_max
            DelATSest_ell = lambda_vec_upsmp[idx_max] + DelATSest_ell_frac/self.p_upsmpFac  # resulting real-valued x_max
            self.SROppm_est_ell = DelATSest_ell / self.B_smpls * 10 ** 6


        # (6) STO-estimation after removing of SRO-induced time offset in CCF-1
        if self.ell >= self.Cont_NumFr + (self.InvShiftFactor_NumFr - 1) + self.AddContWait_NumFr + self.SettlingCSD2avg_NumFr:
            # a) phase shifting of GCSD-1 to remove SRO-induced time offset
            timeOffset_forShift = self.SROppm_est_ell * 10 ** (-6) * self.FFTshift_dxcp * (self.ell - 1)
            idx = np.arange(self.FFTsize_dxcp).transpose()
            expTerm = np.power(math.e, 1j * 2 * math.pi / self.FFTsize_dxcp * timeOffset_forShift * idx)
            GCSD1_smShft = self.GCSD_PhaT_avg * expTerm
            # b) remove components w.o. coherent components
            GCSD1_smShft_ifft = GCSD1_smShft
            # set lower frequency bins (w.o. coherent components) to 0
            GCSD1_smShft_ifft[np.arange(self.LowFreq_InpSig_fl_bin),] = 0
            GCSD1_smShft_ifft[np.arange(self.FFTsize_dxcp - self.LowFreq_InpSig_fl_bin + 1, self.FFTsize_dxcp),] = 0
            # set upper frequency bins (w.o. coherent components) to 0
            GCSD1_smShft_ifft[np.arange(self.FFT_Nyq - self.NyqDist_fu_bin - 1, self.FFT_Nyq + self.NyqDist_fu_bin),] = 0
            # c) go into the time domain via calculation of shifted GCC-1
            GCCF1_sroComp_big = np.fft.fftshift(np.real(np.fft.ifft(GCSD1_smShft_ifft, n=self.FFTsize_dxcp)))
            GCCF1_sroComp = GCCF1_sroComp_big[np.arange(self.FFT_Nyq - self.Upsilon - 1, self.FFT_Nyq + self.Upsilon),]
            # d) averaging over time and zero-phase filtering within the frame (if necessary)
            if self.flag_initiated == False:
                self.GCCF1_smShftAvg[:, 0] = GCCF1_sroComp
            else:
                self.GCCF1_smShftAvg[:, 0] = self.SmoConst_SSOest_alpha * self.GCCF1_smShftAvg[:, 0] + (1 - self.SmoConst_SSOest_alpha) * GCCF1_sroComp

            GCCF1_smShftAvgAbs = np.abs(self.GCCF1_smShftAvg)
            # e) Maximum search over averaged filtered shifted GCC-1 (with real-valued SSO estimates)
            idx_max = GCCF1_smShftAvgAbs.argmax(0)
            if (idx_max == 0) or (idx_max == 2 * self.Upsilon):
                SSOsmp_est_ell_frac = 0
                self.SSOsmp_est_ell = idx_max[0] - self.Upsilon + SSOsmp_est_ell_frac  # resulting real-valued x_max
            else:
                # set supporting points for search of real-valued maximum
                sup_pnts = GCCF1_smShftAvgAbs[
                    np.arange(idx_max - 1, idx_max + 2)]  # supporting points y(x) for x={-1,0,1}
                # calculate fractional of the maximum via x_max=-b/2/a for y(x) = a*x^2 + b*x + c
                SSOsmp_est_ell_frac = (sup_pnts[2, ] - sup_pnts[0, ]) / 2 / (2 * sup_pnts[1, ] - sup_pnts[2, ] - sup_pnts[0, ])
                self.SSOsmp_est_ell = idx_max[0] - self.Upsilon + SSOsmp_est_ell_frac[0]  # resulting real-valued x_max
                # correct for TDOA
                self.SSOsmp_est_ell = self.SSOsmp_est_ell + tdoa*self.RefSampRate_fs_Hz # !verify sign


        # (8) Update counter of DXCP frames within signal and flag initiation
        self.ell += 1
        if (self.flag_initiated == False):
            self.flag_initiated = True

