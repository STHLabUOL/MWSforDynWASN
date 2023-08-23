'''
Implementation some classes for online signal resampling.
'''

import numpy as np
from scipy.signal.windows import hann, boxcar

class OnlineResampler:
    '''
    Online resampler (base class)
    '''
    def __init__(self, 
                 blockSize=2**11,
                 fftSizeBlocks=2,
                 fftOverlapBlocks=1,
                 fftInterpol=2, #factor of fft interpolation
                 window='hann', # hann or rect
                 bufferReach = [1, 1]
                ):

        self.blockSize = blockSize
        self.fftSizeBlocks=fftSizeBlocks
        self.fftSize = blockSize*fftSizeBlocks*fftInterpol
        self.k = np.fft.fftshift(np.arange(-self.fftSize/2, self.fftSize/2))
        if window == 'hann':
            self.win  = hann(blockSize*fftSizeBlocks, sym=False)
        elif window == 'rect':
            self.win = boxcar(blockSize*fftSizeBlocks, sym=False)
        self.inputBufferCurrentBlockIdx = bufferReach[0] # position of "current" frame in buffer
        self.inputBufferSizeBlocks = fftSizeBlocks + sum(bufferReach)
        self.frameDelay = fftSizeBlocks-1 + bufferReach[1]
        #self.outputBufferOutBlockIdx = fftOverlapBlocks-1
        self.outputBufferShiftBlocks = fftSizeBlocks - fftOverlapBlocks
        # State:
        self.inputBuffer = np.zeros((self.inputBufferSizeBlocks*blockSize,)) # example: prev - current - next - next2
        self.outputBuffer = np.zeros((fftSizeBlocks*blockSize,))
        self.ell = 0
        self.shift = 0
        self._overflowWarned = 'None' #'None'/'Left'/'Right'


    def process(self, signalBlock, sro, sto=0):
        '''
        Process signalBlock and return synchronized block from 2 iterations earlier.
        Input:
            sro (scalar): current SRO in ppm
            sto (scalar): current STO in smp 
        Output:
            Synchronized signal block from 2 iterations earlier.
        '''
        # Insert frame in buffer
        self.inputBuffer[:((self.inputBufferSizeBlocks-1)*self.blockSize)] = self.inputBuffer[self.blockSize:]
        self.inputBuffer[(self.inputBufferSizeBlocks-1)*self.blockSize:] = signalBlock
        # Update accumulated shift, separate
        self.shift += sro*1e-6 * self.blockSize
        accShift = self.shift + sto
        integer_shift = np.round(accShift)
        rest_shift = integer_shift - accShift
        # Draw output from buffer: Range from start of second block to end of third, compensated by int shift.
        selectStart = int(self.blockSize*self.inputBufferCurrentBlockIdx + integer_shift)
        selectEnd = selectStart + self.fftSizeBlocks*self.blockSize #int((self.blockSize+2*self.blockSize) + integer_shift)
        # Correct indices in case of overflow
        if selectStart < 0:
            if self._overflowWarned != 'Left':
                print('Warning: Negative shift too large, cannot compensate fully.', str(selectStart))
                self._overflowWarned = 'Left'
            self.shift -= sro*1e-6 * self.blockSize # undo
            selectEnd = selectEnd - selectStart
            selectStart = 0
        elif selectEnd >= np.size(self.inputBuffer):
            print('selectStart: ', selectStart, ', selectEnd:', selectEnd, 'inBufferSize: ', np.size(self.inputBuffer))
            if self._overflowWarned != 'Right':
                print('Warning: Positive shift too large, cannot compensate fully.', str(selectEnd-np.size(self.inputBuffer)))
                self._overflowWarned = 'Right'
            self.shift -= sro*1e-6 * self.blockSize # undo
            selectStart = selectStart - (selectEnd-np.size(self.inputBuffer))
            selectEnd = np.size(self.inputBuffer)
        else:
            self._overflowWarned = 'None'
        selectedBlocks = self.inputBuffer[selectStart:selectEnd]
        # Compensate rest shift via phase shift (fft 2x interpolation)
        selectedBlocks_fft = np.fft.fft(self.win * selectedBlocks, self.fftSize) #fft incl. interpol.
        selectedBlocks_fft *= np.exp(-1j * 2 * np.pi * self.k / self.fftSize * rest_shift)

        # Overlap add (into 2nd and 3rd frame of output buffer)
        self.outputBuffer = self.outputBuffer + np.real(np.fft.ifft(selectedBlocks_fft))[:int(self.fftSizeBlocks*self.blockSize)]
        res = np.copy(self.outputBuffer[:self.blockSize])
        self.outputBuffer[:-(self.blockSize*self.outputBufferShiftBlocks)] = self.outputBuffer[(self.blockSize*self.outputBufferShiftBlocks):]
        self.outputBuffer[-(self.blockSize*self.outputBufferShiftBlocks):] = np.zeros((self.blockSize,))

        return res


'''
RESAMPLER PRESETS
'''

class OnlineResampler_STFT(OnlineResampler):

    '''
    Note: Using non-default arguments may lead to unexpected behaviour
    '''

    defaultParams = {
        'blockSize': 2**11,
        'fftSizeBlocks': 1,
        'fftOverlapBlocks': 0,
        'fftInterpol': 2,
        'window': 'rect',
        'bufferReach': [1, 1]
    }

    def __init__(self,
                blockSize=defaultParams['blockSize'],
                fftSizeBlocks=defaultParams['fftSizeBlocks'],
                fftOverlapBlocks=defaultParams['fftOverlapBlocks'],
                fftInterpol=defaultParams['fftInterpol'],
                window=defaultParams['window'],
                bufferReach = defaultParams['bufferReach']
        ):
        super().__init__(
                blockSize,
                fftSizeBlocks,
                fftOverlapBlocks,
                fftInterpol,
                window,
                bufferReach
        )


class OnlineResampler_OA(OnlineResampler):

    '''
    Note: Using non-default arguments may lead to unexpected behaviour
    '''

    defaultParams = {
        'blockSize': 2**11,
        'fftSizeBlocks': 2,
        'fftOverlapBlocks': 1,
        'fftInterpol': 2,
        'window': 'hann',
        'bufferReach': [1, 1]
    }

    def __init__(self,
                blockSize=defaultParams['blockSize'],
                fftSizeBlocks=defaultParams['fftSizeBlocks'],
                fftOverlapBlocks=defaultParams['fftOverlapBlocks'],
                fftInterpol=defaultParams['fftInterpol'],
                window=defaultParams['window'],
                bufferReach = defaultParams['bufferReach']
        ):
        super().__init__(
                blockSize,
                fftSizeBlocks,
                fftOverlapBlocks,
                fftInterpol,
                window,
                bufferReach
        )
