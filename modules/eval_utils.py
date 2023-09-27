import numpy as np
import scipy
from modules.online_resampler import OnlineResampler_OA

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
        ssnr_async_ = 10*np.log10(var_sync/var_diff_async) if var_diff_async > 0 else np.inf
        ssnr_async.append(ssnr_async_)


    if verbose: print('SSNR: ', ssnr); print('SSNR_async: ', ssnr_async)


    return rmse_t, rmse, ssnr, ssnr_async, amsc, amsc_async, Tc
