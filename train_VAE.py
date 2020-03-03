#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:40:36 2019

@author: smostafa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:19:55 2019

@author: smostafa
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch import optim
from TCD_TIMIT import TIMITnpy, TIMIT, TIMITrawlip
import matplotlib.pyplot as plt
from AV_VAE import VAE, myVAE, myVAErawlip
import os
from torchvision.utils import save_image

from pytorchtools import EarlyStopping

from img_transforms import *
import argparse
from scipy.special import logsumexp
import random

def to_img(x, crop_size):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, crop_size[0], crop_size[1])
    x =np.transpose(x, (0, 1, 3, 2))
    
    return x

crop_size = [67,67]


class NormalLogProb(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, mu, std, z):
    var = torch.pow(std, 2)
    return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - mu, 2) / (2 * var)


def compute_nll(val_dataloader, n_samples, model):
    
    total_log_p_s = 0.0
    total_elbo = 0.0
    
    log_q_zF = NormalLogProb()
    log_p_zF = NormalLogProb()
    log_p_szF = NormalLogProb()
    
    model.eval()
    device = 'cuda'
    
    with torch.no_grad():
            
        for batch_idx, (s, v) in enumerate(val_dataloader):
            
            s = s.to(device)
            v = v.to(device)
            
            # encode "s" and "v" to obtain "z" posterior
            mu_z, logvar_z = model.encode(s, v)
            std_z = torch.exp(0.5*logvar_z)
            eps = torch.randn((mu_z.shape[0], n_samples, mu_z.shape[-1]), device=mu_z.device)
            z = mu_z[:,np.newaxis,:] + std_z[:,np.newaxis,:] * eps  # reparameterization
            # decode "z" to estimate "s"
            std_s = torch.sqrt(model.decode(z, v[:,np.newaxis,:])) 
            
            log_q_z = log_q_zF(mu_z[:,np.newaxis,:], std_z[:,np.newaxis,:], z).sum(-1, keepdim=True)
            log_p_z = log_p_zF(0*z, 1*z, z).sum(-1, keepdim=True)
            log_p_sz = log_p_szF(0*std_s, std_s, s[:,np.newaxis,:]).sum(-1, keepdim=True)
            log_p_s_and_z = log_p_sz + log_p_z
            # importance sampling of approximate marginal likelihood with q(z)
            # as the proposal, and logsumexp in the sample dimension
            elbo = log_p_s_and_z - log_q_z
            
            log_p_s = logsumexp(elbo.cpu().numpy(), axis=1) - np.log(n_samples)
            # average over sample dimension, sum over minibatch
            total_elbo += elbo.cpu().numpy().mean(1).sum()
            # sum over minibatch
            total_log_p_s += log_p_s.sum()
        
    n_data = len(val_dataloader)
    
    return total_elbo / n_data, total_log_p_s / n_data
    
    
def main(args):
    
    activationv = nn.ReLU() #torch.tanh #torch.tanh #nn.ReLU() #torch.tanh #nn.ReLU() #ReLU() SELU(): bon
    #%% Torch seed:
    
    #torch.manual_seed(23)
    
    #%% network parameters
    
    input_dim = 513
    latent_dim = 32   
    hidden_dim_encoder = [128]
    activation = torch.tanh #nn.ReLU() #nn.ReLU() #torch.tanh
    #activation = nn.Softsign()
    
    #%% STFT parameters
    # To use Kaldi features, just 1. change hop_percent to 0.15625 and 2. Change 'LipF.npy' to 'LipKfeats.npy' in file_list
    
    wlen_sec=64e-3
    hop_percent= 0.521 #0.15625, 0.521 #0.25
    fs=16000
    zp_percent=0
    trim=True
    verbose=False
    
    #%% training parameters
    
    
    data_dir_tr = '/local_scratch/smostafa/NTCD_TIMIT_dataset/training_speech/'
    data_dir_val = '/local_scratch/smostafa/NTCD_TIMIT_dataset/validation_speech/'
    
    
    
    
    file_list_tr = [os.path.join(root, name)
             for root, dirs, files in os.walk(data_dir_tr)
             for name in files
             if name.endswith('.wav')] 
    
    file_list_val = [os.path.join(root, name)
             for root, dirs, files in os.walk(data_dir_val)
             for name in files
             if name.endswith('.wav')] 
    
    
    #file_list_val = sorted(file_list_val)
    
    
    #random.shuffle(file_list_tr)
#    random.shuffle(file_list_val)
#    file_list_val = file_list_val[:10]
    
    #print(file_list_tr[0:50])
    #print(file_list_val[0:50])
    #file_list_val = sorted(file_list_val)
    
    #random.shuffle(file_list)
    
    #file_list = file_list[:1000]
    print(file_list_tr[0])
    print(file_list_val[0])
    
    print('Number of training samples: ', len(file_list_tr))
    print('Number of validation samples: ', len(file_list_val))
    print('done')
    
    lr = args.lr
    epoches = 200
    batch_size = 128
    save_dir = './saved_model_tune'
    num_workers = 0
    shuffle_file_list = True
    shuffle_samples_in_batch = True

    
    device = 'cuda'
    
    #saved_path_pretrained = '/scratch/bootes/smostafa/AV_VAE/audiovisual-vae/VAE_pytorch/saved_model/final_models/init_model.pt'
    
    # check Pretraining
    vae_mode = args.vae_mode # 'myA_lr4tanh_ES' #myAVenc_lr3tanh myAVencdecFreezZdec
    blockVenc = args.blockVenc
    blockVdec = args.blockVdec
    blockZ = args.blockZ
    pgrad = args.pgrad 
    lam = args.lam    
    vae_joint = args.vae_joint

    aux_video = False
    if lam > 0:
       aux_video = True

       
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
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!! x_block !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!! x_block !!!!!!!!!!!!!!!!!!!!!!!!!!!
    x_block = 0.0
    # !!!!!!!!!!!!!!!!!!!!!!!!!!! x_block !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!! x_block !!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    vae = myVAE(input_dim=input_dim, latent_dim=latent_dim, 
            hidden_dim_encoder=hidden_dim_encoder, batch_size=batch_size, 
            activation=activation, activationv=activationv, blockZ = blockZ, blockVenc = blockVenc, blockVdec = blockVdec, aux_video = aux_video, vae_joint = vae_joint, x_block = x_block).to(device)
    
    #%% Pretraining:
    
    #vae.load_state_dict(torch.load(saved_path_pretrained))
    
    
    print('vae model')
    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    print('optimizer')
    # loss function
    if blockVenc and blockZ:
        def loss_function(recon_xi, xi, mui, logvari):  
            recon = torch.sum( torch.log(recon_xi+0e-5) + xi/(recon_xi+0e-5) ) 
            return recon
    # !!!!!!!!!!!!!!!!!!!!!!!!!!! x_block !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!! x_block !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Loss function changed!!! No KL term!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!! x_block !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!! x_block !!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    else:
        def loss_function(recon_xi, xi, mui, logvari):  
            recon = torch.sum( torch.log(recon_xi+0e-5) + xi/(recon_xi+0e-5) ) 
            KLD = -0.5 * torch.sum(logvari - mui.pow(2) - logvari.exp())
            return recon + 1*KLD
    
    def loss_function_aux(recx, x_, mu_, logvar_, v, vaux):

         recon = torch.sum( torch.log(recx)+x_/(recx))+lam*torch.sum((v-vaux)**2)
         KLD = -0.5 * torch.sum(logvar_ - mu_.pow(2) - logvar_.exp())
         return recon + KLD

    def loss_function_joint(recx, x_, mu_, logvar_, mu_j, logvar_j, v):

         reconA = torch.sum( torch.log(recx)+x_/(recx))
         reconV = 0.5*torch.sum(logvar_j+((v-mu_j)**2)/logvar_j.exp())
         KLD = -0.5 * torch.sum(logvar_ - mu_.pow(2) - logvar_.exp())
         return reconA + reconV + KLD, reconA+KLD

    def SwitchGradParam(vae, gmode = False):
        
        if blockVdec and not blockVenc:
            vae.encoder_layerX.bias.requires_grad = gmode
            vae.encoder_layerX.weight.requires_grad = gmode
            return vae
            
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
    
    #loss_function = nn.MSELoss(size_average = False)

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
    
    print('aux_video', aux_video)
    print('vae_joint', vae_joint)
    
    for epoch in range(epoch0, epoches):
        
        vae.train()
    
        for batch_idx, (batch_audio, batch_video) in enumerate(train_dataloader):
            # toggle requires.grad occasionally:
            rn = np.random.rand()
            if rn < pgrad:
                vae = SwitchGradParam(vae, gmode = False)
            # Data augmentation and mean/std normalization
    #        batch_imgrc = RandomCrop(batch_video.numpy(), (61,61))
    #        batch_imgcn = ColorNormalize(batch_imgrc)
    #        batch_imghp = HorizontalFlip(batch_imgcn, 0)
    #        batch_video = torch.from_numpy(batch_imgcn.astype(np.float32))                   
            batch_audio = batch_audio.to(device)
            batch_video = batch_video.to(device)
            if aux_video:
               recon_batch, mu, logvar, vaux = vae(batch_audio, batch_video)
               loss = loss_function_aux(recon_batch, batch_audio, mu, logvar, batch_video, vaux)
               vorig = batch_video.cpu().data
            elif vae_joint:
               recon_batch, mu, logvar, mu_j0, logvar_j0 = vae(batch_audio, batch_video)
               loss, lossV = loss_function_joint(recon_batch, batch_audio, mu, logvar, mu_j0, logvar_j0, batch_video)
           

            else:
               recon_batch, mu, logvar = vae(batch_audio, batch_video)
               loss = loss_function(recon_batch, batch_audio, mu, logvar)

            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            vae = SwitchGradParam(vae, gmode = True)
        
        
    
        
        if epoch % skip_epoch == 0:
    
            # validation loss:
            vae.eval()
            
#            a, b = compute_nll(val_dataloader, 100, vae)
#            print(a,b)
#            tot_elbo = np.append(tot_elbo, a)
#            tot_logp = np.append(tot_logp, b)
#            
#            np.save(os.path.join(save_dir, 'Elbo_'+str(vae_mode)), tot_elbo)
#            np.save(os.path.join(save_dir, 'Logp_'+str(vae_mode)), tot_logp)
            
            with torch.no_grad():
                for batch_idx, (batch_audio, batch_video) in enumerate(val_dataloader):
                    # Data center-cropping and mean/std normalization
    #                batch_imgcc = CenterCrop(batch_video.numpy(), (61,61))
    #                batch_imgcn = ColorNormalize(batch_imgcc)
    #                batch_video = torch.from_numpy(batch_imgcn.astype(np.float32))                   
                    batch_audio = batch_audio.to(device)
                    batch_video = batch_video.to(device)
                    if aux_video:
                        recon_batch, mu, logvar, _= vae(batch_audio, batch_video)
                    elif vae_joint:
                        recon_batch, mu, logvar, mu_j1, logvar_j1 = vae(batch_audio, batch_video)
                    else:
                        recon_batch, mu, logvar= vae(batch_audio, batch_video)
                        
                    loss = loss_function(recon_batch, batch_audio, mu, logvar)  
                    valid_losses.append(loss.item())      
#                    a, b = compute_nll(batch_audio, batch_video, 100,  vae)
#                    print(a,b)
            
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
    
        
#        if epoch % 5 == 0 and aux_video:
#            
#            recon_batchC = vaux.cpu().data
#            saveRec = to_img(recon_batchC, crop_size)
#            save_image(saveRec, './saved_model_tune/image_{}_Rec.png'.format(epoch))
#            save = to_img(vorig, crop_size)
#            save_image(save, './saved_model_tune/image_{}.png'.format(epoch))
            
    save_file = os.path.join(save_dir, 'final_model_'+str(vae_mode)+'.pt')
    torch.save(vae.state_dict(), save_file)

#%%
 

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--blockVenc", type=float, default=1.0)
    parser.add_argument("--blockVdec", type=float, default=1.0)
    parser.add_argument("--blockZ", type=float, default=0.0)
    parser.add_argument("--pgrad", type=float, default=-0.2)
    parser.add_argument("--vae_mode", type=str, default='a_vae', help='name of the used vae net')
    parser.add_argument("--vae_joint", action = 'store_true', help='name of the used vae net')
    parser.add_argument("--lam", type=float, default = 0.0)
    
    args = parser.parse_args()

    main(args)
 
    
    
    
    
    
    

def plot_train_val_losses(vae_mode):
       
    save_dir = '/local_scratch/smostafa/AV_VAE/audiovisual-vae/VAE_pytorch/saved_model_tune/'
    save_loss_dir_tr = os.path.join(save_dir, 'Train_loss_'+str(vae_mode))  
    save_loss_dir_val = os.path.join(save_dir, 'Validation_loss_'+str(vae_mode)) 

    print(save_loss_dir_tr)
    avg_train_lossess = np.load(save_loss_dir_tr+str('.npy'))   
    avg_valid_lossess = np.load(save_loss_dir_val+str('.npy'))   
    
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(avg_train_lossess[0:])+1),avg_train_lossess[0:], range(1,len(avg_valid_lossess[0:])+1),avg_valid_lossess[0:], linewidth=2.0)


#    # find position of lowest validation loss
    minposs = np.argmin(avg_valid_lossess)+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    plt.gca().legend(('Training loss','Validation loss','Early Stopping Checkpoint' ), prop={'size': 18})
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.title(str(vae_mode[:-3])+'----'+str(np.amin(avg_train_lossess)), fontsize=18)
    plt.grid(True)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join('/local_scratch/smostafa/plots_av_vae',str(vae_mode)+'loss_plot.png'), bbox_inches='tight')
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#for idf , file in  enumerate(file_list):
#    print('processing {}'.format(idf))
#    os.remove(file)

#        # training loss:
#        vae.eval()
#        with torch.no_grad():
#            for batch_idx, (batch_audio, batch_video) in enumerate(train_dataloader):
#                batch_audio = batch_audio.to(device)
#                batch_video = batch_video.to(device)
#                recon_batch, mu, logvar = vae(batch_audio, batch_video)
#                loss = loss_function(recon_batch, batch_audio, mu, logvar)
#                train_losses.append(loss.item())
    
    
#%%
    
#Saving & Loading a General Checkpoint for Inference and/or Resuming Training
#
#Save:
#torch.save({
#            'epoch': epoch,
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': loss,
#            ...
#            }, PATH)
#Load:
#model = TheModelClass(*args, **kwargs)
#optimizer = TheOptimizerClass(*args, **kwargs)
#
#checkpoint = torch.load(PATH)
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']
#loss = checkpoint['loss']
#
#model.eval()
## - or -
#model.train()
