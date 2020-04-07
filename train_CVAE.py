#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:19:50 2019

@author: smostafa

"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch import optim
from TCD_TIMIT import TIMIT
import matplotlib.pyplot as plt
from AV_VAE import CVAERTied2
import os
import sys
from pytorchtools import EarlyStopping
import argparse

#%% Torch seed:

#torch.manual_seed(23)

#%% network parameters
def main(args):
    
    input_dim = 513
    latent_dim = 32 
    hidden_dim_encoder = [128]
    activation = torch.tanh  # activation for audio nets
    activationV = nn.ReLU() # activation for video nets
    
    #%% STFT parameters
    
    wlen_sec=64e-3
    hop_percent= 0.521 
    fs=16000
    zp_percent=0
    trim=False
    verbose=False
    
    #%% training parameters
    
    
    data_dir_tr = '/local_scratch/smostafa/NTCD_TIMIT_dataset/training_speech/'
    data_dir_val = '/local_scratch/smostafa/NTCD_TIMIT_dataset/validation_speech/'
    
    if not os.path.isdir(data_dir_tr):
        data_dir_tr = '/local_scratch/data/perception/smostafa/NTCD_TIMIT_dataset/training_speech/'
        data_dir_val = '/local_scratch/data/perception/smostafa/NTCD_TIMIT_dataset/validation_speech/'    
    
    file_list_tr = [os.path.join(root, name)
             for root, dirs, files in os.walk(data_dir_tr)
             for name in files
             if name.endswith('.wav')] 
    
    file_list_val = [os.path.join(root, name)
             for root, dirs, files in os.walk(data_dir_val)
             for name in files
             if name.endswith('.wav')] 
    
    
    print(file_list_tr[0])
    print(file_list_val[0])
    
    print('Number of training samples: ', len(file_list_tr))
    print('Number of validation samples: ', len(file_list_val))
    print('done')
    
    lr = 1e-4
    epoches = 200
    batch_size = 128
    save_dir = './saved_model_tune'
    num_workers = 0
    shuffle_file_list = True
    shuffle_samples_in_batch = True

    
    device = 'cuda'
        
    vae_mode = args.vae_mode 
    pgrad = args.pgrad 
    beta = args.beta 
    
    #%%
    
    # create training dataloader 
    train_dataset = TIMIT('training', file_list=file_list_tr, wlen_sec=wlen_sec, 
                     hop_percent=hop_percent, fs=fs, zp_percent=zp_percent, 
                     trim=trim, verbose=verbose, batch_size=batch_size, 
                     shuffle_file_list=shuffle_file_list)
    
    # torch load will call __getitem__ of TIMIT to create Batch by randomly 
    # (if shuffle=True) selecting data sample.
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, 
                                       shuffle=shuffle_samples_in_batch, 
                                       num_workers=num_workers)
    
    
    # create validation dataloader 
    val_dataset = TIMIT('validation', file_list=file_list_val, wlen_sec=wlen_sec, 
                     hop_percent=hop_percent, fs=fs, zp_percent=zp_percent, 
                     trim=trim, verbose=verbose, batch_size=batch_size, 
                     shuffle_file_list=shuffle_file_list)
    
    # torch load will call __getitem__ of TIMIT to create Batch by randomly 
    # (if shuffle=True) selecting data sample.
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, 
                                       shuffle=shuffle_samples_in_batch, 
                                       num_workers=num_workers)
    
    print('data loader')
    print('len(train_dataloader.dataset)', len(train_dataloader.dataset))
    print('len(val_dataloader.dataset)', len(val_dataloader.dataset))
    # init model
    
    
    vae = CVAERTied2(input_dim=input_dim, latent_dim=latent_dim, 
            hidden_dim_encoder=hidden_dim_encoder, batch_size=batch_size, 
            activation=activation, activationV = activationV).to(device)
    
    #%% Pretraining:
    
    #saved_path_pretrained = '/scratch/bootes/smostafa/AV_VAE/audiovisual-vae/VAE_pytorch/saved_model/final_models/init_model.pt'
    
    #vae.load_state_dict(torch.load(saved_path_pretrained))

    print('vae model')
    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    print('optimizer')
    # loss function
    
    def loss_function(recon_xi, xi, mui, logvari, muzpi, logvarzpi, recon_xi_zp):  
        recon = torch.sum( torch.log(recon_xi) + xi/(recon_xi) ) 
        KLD = -0.5 * torch.sum(logvari-logvarzpi - (logvari.exp()+(mui-muzpi).pow(2))/(logvarzpi.exp()))
        recon_zp = torch.sum( torch.log(recon_xi_zp) + xi/(recon_xi_zp) ) 
        
        return beta*(recon + KLD)+(1.-beta)*recon_zp
    
    
    def SwitchGradParam(vae, gmode = False):
                
        vae.latent_logvar_layer.bias.requires_grad = gmode
        vae.latent_logvar_layer.weight.requires_grad = gmode
        vae.latent_mean_layer.bias.requires_grad = gmode
        vae.latent_mean_layer.weight.requires_grad = gmode
        vae.decoder_layerZ.bias.requires_grad = gmode
        vae.decoder_layerZ.weight.requires_grad = gmode
        vae.encoder_layerX.bias.requires_grad = gmode
        vae.encoder_layerX.weight.requires_grad = gmode
        
        return vae
    
    #%% main loop for training
        
    save_loss_dir_tr = os.path.join(save_dir, 'Train_loss_'+str(vae_mode))  
    save_loss_dir_val = os.path.join(save_dir, 'Validation_loss_'+str(vae_mode)) 
            
    # initialize the early_stopping object
    checkpoint_path = os.path.join(save_dir, str(vae_mode)+'_checkpoint.pt')
    
    early_stopping = EarlyStopping(save_dir = checkpoint_path)
    
      
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
        
    skip_epoch = 1
    

    epoch0 = 0
    
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        vae.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch0 = checkpoint['epoch']
        
        print("=> loaded checkpoint '{}' (epoch {})"
          .format(checkpoint_path, checkpoint['epoch']))
    
    if os.path.isfile(save_loss_dir_tr+'.npy'):
        avg_train_losses = np.load(save_loss_dir_tr+'.npy')
    
    if os.path.isfile(save_loss_dir_val+'.npy'):
        avg_valid_losses = np.load(save_loss_dir_val+'.npy')
    
    print('beta is ...', beta)
    
    for epoch in range(epoch0, epoches):
        
        vae.train()
    
        for batch_idx, (batch_audio, batch_video) in enumerate(train_dataloader):
            # toggle requires.grad occasionally:
            rn = np.random.rand()
            if rn < pgrad:
                vae = SwitchGradParam(vae, gmode = False)
                
            batch_audio = batch_audio.to(device)
            batch_video = batch_video.to(device)
            recon_batch, mu, logvar, muz, logvarz, recon_batch_zp = vae(batch_audio, batch_video)
            loss = loss_function(recon_batch, batch_audio, mu, logvar, muz, logvarz, recon_batch_zp)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            vae = SwitchGradParam(vae, gmode = True)
        
        
        if epoch % skip_epoch == 0:
    
            # validation loss:
            vae.eval()
            with torch.no_grad():
                for batch_idx, (batch_audio, batch_video) in enumerate(val_dataloader):
                    batch_audio = batch_audio.to(device)
                    batch_video = batch_video.to(device)
                    recon_batch, mu, logvar, muz, logvarz, recon_batch_zp = vae(batch_audio, batch_video)
                    loss = loss_function(recon_batch, batch_audio, mu, logvar, muz, logvarz, recon_batch_zp)
                    valid_losses.append(loss.item())      
                    
                    
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.sum(train_losses) / len(train_dataloader.dataset)
        valid_loss = np.sum(valid_losses) / len(val_dataloader.dataset)
        
        avg_train_losses = np.append(avg_train_losses, train_loss)
        avg_valid_losses = np.append(avg_valid_losses, valid_loss)
           
        epoch_len = len(str(epoches))
        
        print_msg = (f'====> Epoch: [{epoch:>{epoch_len}}/{epoches:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
             
        np.save(save_loss_dir_tr, avg_train_losses)
        np.save(save_loss_dir_val, avg_valid_losses)
            
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
            
    
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(train_loss, valid_loss, vae, epoch, optimizer)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    
    save_file = os.path.join(save_dir, 'final_model_'+str(vae_mode)+'.pt')
    torch.save(vae.state_dict(), save_file)

#%%

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--pgrad", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=1.0, help='weight between log p(x|z) and log p(x|zp)')
    parser.add_argument("--vae_mode", type=str, default='a_vae', help='name of the used vae net')

    
    args = parser.parse_args()

    main(args)
