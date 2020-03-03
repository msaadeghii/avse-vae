#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Mostafa Sadeghi (mostafa.sadeghi@inria.fr)
License agreement in LICENSE.txt
"""

import torch
import numpy as np
import soundfile as sf 
import librosa
import torch.nn as nn

from MCEM_algo import MCEM_algo, MCEM_algo_cvae

from AV_VAE import myVAE, CVAERTied, myDecoder, CDecoderRTied
import os


#%% network parameters

input_dim = 513
latent_dim = 32
device = 'cpu' # 'cuda' 
hidden_dim_encoder = [128]
activation = torch.tanh
activationv = nn.ReLU()
landmarks_dim = 67*67 # if you use raw video data, this dimension should be 67*67. Otherwise, if you use the
#                       pre-trained ASR feature extractor, this dimension is 1280

#%% MCEM algorithm parameters

niter_MCEM = 200 # number of iterations for the MCEM algorithm
niter_MH = 40 # total number of samples for the Metropolis-Hastings algorithm
burnin = 30 # number of initial samples to be discarded
var_MH = 0.01 # variance of the proposal distribution
tol = 1e-5 # tolerance for stopping the MCEM iterations
    
    
#%% STFT parameters

wlen_sec=64e-3
hop_percent= 0.521 
   
  
fs=16000
wlen = int(wlen_sec*fs) # window length of 64 ms
wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
  
wlen = int(wlen_sec*fs) # window length of 64 ms
wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
nfft = wlen
hop = np.int(hop_percent*wlen) # hop size
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window
 
save_dir = './results'  # directory to save results
saved_models = './models' # where the VAE model is
mix_file = './data/mix.wav'  # input noisy speech
video_file = './data/video.npy' # input video data

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
#%% Read input audio and video observations:

x, fs = librosa.load(mix_file, sr=None)
x = x/np.max(np.abs(x)) # normalize input mixture
v = np.load(video_file)     
T_orig = len(x)


K_b = 10 # NMF rank for noise model

X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)


# Observed power spectrogram of the noisy mixture signal
X_abs_2 = np.abs(X)**2
X_abs_2_tensor = torch.from_numpy(X_abs_2.astype(np.float32))

F, N = X.shape

# check if the number of video frames is equal to the number of spectrogram frames. If not, augment video by repeating the last frame:
Nl = np.maximum(N, v.shape[1])

if v.shape[1] < Nl:
    v = np.hstack((v, np.tile(v[:, [-1]], Nl-v.shape[1])))
  
v = v.T
v = torch.from_numpy(v.astype(np.float32))
v.requires_grad = False


# Random initialization of NMF parameters
eps = np.finfo(float).eps
np.random.seed(0)
W0 = np.maximum(np.random.rand(F,K_b), eps)
H0 = np.maximum(np.random.rand(K_b,N), eps)


V_b0 = W0@H0
V_b_tensor0 = torch.from_numpy(V_b0.astype(np.float32))

# All-ones initialization of the gain parameters
g0 = np.ones((1,N))
g_tensor0 = torch.from_numpy(g0.astype(np.float32))

#%% Here, we test the performance of audio-only VAE. For that, we need to set blockVenc = 1 and blockVdec = 1 such that
#   the path from visual data in the encoder and decoder is blocked.

saved_model_a_vae = os.path.join(saved_models, 'A_VAE_checkpoint.pt')  
# Loading the pre-trained model:  
vae = myVAE(input_dim = input_dim, latent_dim = latent_dim, hidden_dim_encoder = hidden_dim_encoder,
            activation = activation, activationv = activationv,
            blockZ = 0., blockVenc = 1., blockVdec = 1.,
            x_block = 0., landmarks_dim = 1280).to(device)

checkpoint = torch.load(saved_model_a_vae, map_location = 'cpu') 
vae.load_state_dict(checkpoint['model_state_dict'], strict = False)
decoder = myDecoder(vae)

# As we do not train the models, we set them to the "eval" mode:
vae.eval()
decoder.eval()

v0 = np.zeros((Nl, 1280))
v0 = torch.from_numpy(v0.astype(np.float32))
v0.requires_grad = False

# we will not update the network parameters
for param in decoder.parameters():
    param.requires_grad = False

# Initialize the latent variables by encoding the noisy mixture 
with torch.no_grad():
    data_orig = X_abs_2
    data = data_orig.T
    data = torch.from_numpy(data.astype(np.float32))
    data = data.to(device)
    vae.eval()
    z, _ = vae.encode(data, v0)
    z = torch.t(z)
    
Z_init = z.numpy()    
# Instanciate the MCEM algo
mcem_algo = MCEM_algo(X=X, W=W0, H=H0, Z=Z_init, v = v0, decoder=decoder,
                      niter_MCEM=niter_MCEM, niter_MH=niter_MH, burnin=burnin,
                      var_MH=var_MH)

# Run the MCEM algo
cost, niter_final = mcem_algo.run(hop=hop, wlen=wlen, win=win, tol=tol)

# Separate the sources from the estimated parameters
mcem_algo.separate(niter_MH=100, burnin=75)

s_hat = librosa.istft(stft_matrix=mcem_algo.S_hat, hop_length=hop,
                      win_length=wlen, window=win, length=T_orig)
b_hat = librosa.istft(stft_matrix=mcem_algo.N_hat, hop_length=hop,
                      win_length=wlen, window=win, length=T_orig)


# save the results:
save_vae = os.path.join(save_dir, 'A-VAE')
if not os.path.isdir(save_vae):
    os.makedirs(save_vae)
    
sf.write(os.path.join(save_vae,'est_speech.wav'), s_hat, fs)
sf.write(os.path.join(save_vae,'est_noise.wav'), b_hat, fs)

print('A-VAE finished ...')

#%% Here, we test the performance of audio-visual VAE where the prior for z is standard Gaussian. For that, we need to set blockVenc = 0., blockVdec = 0.
#   to allow the visual information go through the encoder and decoder
  
saved_model_av_vae = os.path.join(saved_models, 'AV_VAE_checkpoint.pt') 
# Loading the pre-trained model:  
vae = myVAE(input_dim = input_dim, latent_dim = latent_dim, hidden_dim_encoder = hidden_dim_encoder,
            activation = activation, activationv = activationv,
            blockZ = 0., blockVenc = 0., blockVdec = 0.,
            x_block = 0., landmarks_dim = landmarks_dim).to(device)

checkpoint = torch.load(saved_model_av_vae, map_location = 'cpu') 
vae.load_state_dict(checkpoint['model_state_dict'], strict = False)
decoder = myDecoder(vae)

# As we do not train the models, we set them to the "eval" mode:
vae.eval()
decoder.eval()


# we will not update the network parameters
for param in decoder.parameters():
    param.requires_grad = False

# Initialize the latent variables by encoding the noisy mixture 
with torch.no_grad():
    data_orig = X_abs_2
    data = data_orig.T
    data = torch.from_numpy(data.astype(np.float32))
    data = data.to(device)
    vae.eval()
    z, _ = vae.encode(data, v)
    z = torch.t(z)
    
Z_init = z.numpy()    
# Instanciate the MCEM algo
mcem_algo = MCEM_algo(X=X, W=W0, H=H0, Z=Z_init, v = v, decoder=decoder,
                      niter_MCEM=niter_MCEM, niter_MH=niter_MH, burnin=burnin,
                      var_MH=var_MH)

# Run the MCEM algo
cost, niter_final = mcem_algo.run(hop=hop, wlen=wlen, win=win, tol=tol)

# Separate the sources from the estimated parameters
mcem_algo.separate(niter_MH=100, burnin=75)

s_hat = librosa.istft(stft_matrix=mcem_algo.S_hat, hop_length=hop,
                      win_length=wlen, window=win, length=T_orig)
b_hat = librosa.istft(stft_matrix=mcem_algo.N_hat, hop_length=hop,
                      win_length=wlen, window=win, length=T_orig)


# save the results:
save_vae = os.path.join(save_dir, 'AV-VAE')
if not os.path.isdir(save_vae):
    os.makedirs(save_vae)
    
sf.write(os.path.join(save_vae,'est_speech.wav'), s_hat, fs)
sf.write(os.path.join(save_vae,'est_noise.wav'), b_hat, fs)

print('AV-VAE finished ...')    
    
#%% Here, we test the performance of conditional VAE (CVAE) for audio-visual speech enhancement. The CVAE here contains a feature extractor that is
#   shared among all the visual subnetworks. That's why we put a "Tied" in its name

saved_model_av_cvae = os.path.join(saved_models, 'AV_CVAE_checkpoint.pt')     
# Loading the pre-trained model:
vae = CVAERTied(input_dim = input_dim, latent_dim = latent_dim, hidden_dim_encoder = hidden_dim_encoder,
           activation = activation, activationV = activationv).to(device)

checkpoint = torch.load(saved_model_av_cvae, map_location = 'cpu')
vae.load_state_dict(checkpoint['model_state_dict'], strict = False)
   
decoder = CDecoderRTied(vae)


vae.eval()
decoder.eval()


# we will not update the network parameters
for param in decoder.parameters():
    param.requires_grad = False

# Initialize the latent variables by encoding the noisy mixture
    
with torch.no_grad():
    data_orig = X_abs_2
    data = data_orig.T
    data = torch.from_numpy(data.astype(np.float32))
    data = data.to(device)
    vae.eval()
    z, _ = vae.encode(data, v)
    z = torch.t(z)
    mu_z, logvar_z = vae.zprior(v)
    
Z_init = z.numpy() 
mu_z = mu_z.numpy()
logvar_z = logvar_z.numpy()

# Instanciate the MCEM algo
mcem_algo = MCEM_algo_cvae(mu_z, logvar_z, X=X, W=W0, H=H0, Z=Z_init, v = v, decoder=decoder,
                      niter_MCEM=niter_MCEM, niter_MH=niter_MH, burnin=burnin, var_MH=var_MH)

# Run the MCEM algo
cost, niter_final = mcem_algo.run(hop=hop, wlen=wlen, win=win, tol=tol)

# Separate the sources from the estimated parameters
mcem_algo.separate( niter_MH=100, burnin=75)

s_hat = librosa.istft(stft_matrix=mcem_algo.S_hat, hop_length=hop,
                      win_length=wlen, window=win, length=T_orig)
b_hat = librosa.istft(stft_matrix=mcem_algo.N_hat, hop_length=hop,
                      win_length=wlen, window=win, length=T_orig)

# save the results:
save_vae = os.path.join(save_dir, 'AV-CVAE')
if not os.path.isdir(save_vae):
    os.makedirs(save_vae)
    
sf.write(os.path.join(save_vae,'est_speech.wav'), s_hat, fs)
sf.write(os.path.join(save_vae,'est_noise.wav'), b_hat, fs)
            
print('AV-CVAE finished ...')        
