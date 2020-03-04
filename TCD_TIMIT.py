#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Mostafa Sadeghi (mostafa.sadeghi@inria.fr)
License agreement in LICENSE.txt
"""

import torch
from torch.utils import data
import os
import numpy as np
import librosa
import random


class TIMIT(data.Dataset):
    """
    Customize a dataset for PyTorch, in order to be used with torch dataloarder,
    at least the three following functions should be defined.
    """

    def __init__(self, data_mode, file_list, wlen_sec=64e-3, 
                 hop_percent=0.25, fs=16000, zp_percent=0, trim=False,
                 verbose=False, batch_size=128, shuffle_file_list=True, video_part = True):
        """
        Initialization of class TIMIT
        """
        super(TIMIT, self).__init__()
        self.batch_size = batch_size
        self.file_list = file_list
        self.data_mode = data_mode
        self.wlen_sec = wlen_sec # STFT window length in seconds
        self.hop_percent = hop_percent  # hop size as a percentage of the window length
        self.fs = fs
        self.zp_percent = zp_percent
        self.wlen = self.wlen_sec*self.fs # window length in samples
        self.wlen = np.int(np.power(2, np.ceil(np.log2(self.wlen)))) # next power of 2
        self.hop = np.int(self.hop_percent*self.wlen) # hop size in samples
        self.nfft = self.wlen + self.zp_percent*self.wlen # number of points of the discrete Fourier transform
        self.win = np.sin(np.arange(.5,self.wlen-.5+1)/self.wlen*np.pi); # sine analysis window
        self.video_part = video_part
        
        self.cpt_file = 0
        self.trim = trim
        self.current_frame = 0
        self.tot_num_frame = 0
        self.verbose = verbose
        self.shuffle_file_list = shuffle_file_list
        self.compute_len()
        
    def compute_len(self):
        
        self.num_samples = 0
        
        for cpt_file, wavfile in enumerate(self.file_list):

            path, file_name = os.path.split(wavfile)
            path, speaker = os.path.split(path)
            path, dialect = os.path.split(path)
            path, set_type = os.path.split(path)
            
            x, fs_x = librosa.load(wavfile, sr = self.fs)
            
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')        
                
            if self.trim: # remove beginning and ending silence
                
                xt, index = librosa.effects.trim(x, top_db=30)
                
                x = np.pad(xt, int(self.nfft // 2), mode='reflect') # (cf. librosa.core.stft)
                
            else:
                
                x = np.pad(x, int(self.nfft // 2), mode='reflect') # (cf. librosa.core.stft)
            
            n_frames = 1 + int((len(x) - self.wlen) / self.hop)
            
            self.num_samples += n_frames

    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return self.num_samples

    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        :param index: index for which we choose from list of dataset, self.file_list
        :return:
        """
        
        if self.current_frame == self.tot_num_frame:
        
            if self.cpt_file == len(self.file_list):
                self.cpt_file = 0
                if self.shuffle_file_list:
                    random.shuffle(self.file_list)
            
            wavfile = self.file_list[self.cpt_file]
            self.cpt_file += 1
            
            path, file_name = os.path.split(wavfile)
            path, speaker = os.path.split(path)
            
            path, dialect = os.path.split(path)
            path_ntcdVal = path
            path, set_type = os.path.split(path)
            path_ntcdTr = path

            x, fs_x = librosa.load(wavfile, sr = self.fs)
            
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')  
            
            x = x/np.max(np.abs(x))
            
            if self.video_part:
                if  self.data_mode == 'training':           
                    # Read video data
                    v = np.load(os.path.join(path_ntcdTr, str(self.data_mode)+'_video' , str(dialect), file_name[:-4]+'Raw.npy')) 
                elif self.data_mode == 'validation':
                    v = np.load(os.path.join(path_ntcdVal, str(self.data_mode)+'_video' , str(speaker), file_name[:-4]+'Raw.npy'))
                else:
                    raise NameError('Wrong "training" or "validation" mode specificed.')
            
            if self.trim: # remove beginning and ending silence
                
                xt, index = librosa.effects.trim(x, top_db=30)
                            
                X = librosa.stft(xt, n_fft=self.nfft, hop_length=self.hop, 
                                 win_length=self.wlen,
                                 window=self.win) # STFT
                
            else:
                
                X = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop, 
                                 win_length=self.wlen,
                                 window=self.win) # STFT                
            
            self.audio_data = np.abs(X)**2 # num_freq x num_frames
            if self.video_part:
                self.video_data = v 
            else:
                self.video_data = self.audio_data.copy()
                
            # check if num of frames equal
            self.current_frame = 0
            self.tot_num_frame = np.minimum(self.audio_data.shape[1],self.video_data.shape[1])
            
        audio_frame = self.audio_data[:,self.current_frame]    
        video_frame = self.video_data[:,self.current_frame]    
        
        self.current_frame += 1
        
        audio_frame = torch.from_numpy(audio_frame.astype(np.float32))
        video_frame = torch.from_numpy(video_frame.astype(np.float32))
        
        return audio_frame, video_frame

