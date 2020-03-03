#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Mostafa Sadeghi (mostafa.sadeghi@inria.fr)
License agreement in LICENSE.txt
"""

import torch
from torch import nn
import torch.nn.functional as F


class myVAE(nn.Module):
    
    
    def __init__(self, input_dim=None, latent_dim=None,
                 hidden_dim_encoder=None, activation=None, activationv=None,
                 blockZ = 1., blockVenc = 1., blockVdec = 1., x_block = 0.0, landmarks_dim = 67*67):
        
        super(myVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.landmarks_dim = landmarks_dim # dimension of the input raw visual data
        self.hidden_dim_encoder = hidden_dim_encoder 
        self.activation = activation # activation for audio layers
        self.activationv = activationv # activation for video layers
        self.x_block = x_block
        
        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])
        self.decoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])
        self.decoder_layerV0 = nn.Linear(self.landmarks_dim, 512)        

        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
        self.encoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])
        self.encoder_layerV0 = nn.Linear(self.landmarks_dim, 512)        

        self.blockZ = blockZ
        self.blockVenc = blockVenc
        self.blockVdec = blockVdec
        
        self.output_layer = nn.Linear(hidden_dim_encoder[0], self.input_dim) 
    
        
        #### Define bottleneck layer ####  
        
        self.latent_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        
                
    def encode(self, x, v):
        ve = self.encoder_layerV0(v)
        ve = self.activationv(ve)
        xv = (1.-self.x_block)*self.encoder_layerX(x) + (1.-self.blockVenc)*self.encoder_layerV(ve)
        he = self.activation(xv)
        
        return self.latent_mean_layer(he), self.latent_logvar_layer(he)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, v):
        vd = self.decoder_layerV0(v)
        vd = self.activationv(vd)
        zv = (1.-self.blockZ)*self.decoder_layerZ(z) + (1.-self.blockVdec)*self.decoder_layerV(vd)
        hd = self.activation(zv)
            
        return torch.exp(self.output_layer(hd))

    def forward(self, x, v):     
        mu, logvar = self.encode(x, v)
        z = self.reparameterize(mu, logvar)

        return self.decode(z, v), mu, logvar
           
  
class myDecoder(nn.Module):
    
    def __init__(self, vae):
        
        super(myDecoder, self).__init__()
        self.latent_dim = vae.latent_dim
        self.activation = vae.activation  
        self.activationv = vae.activationv
        self.output_layer = None
        self.blockVdec = vae.blockVdec
        self.blockZ = vae.blockZ
        self.build(vae)
        
    def build(self, vae):
                        
        self.output_layer = vae.output_layer
        self.decoder_layerZ = vae.decoder_layerZ
        self.decoder_layerV = vae.decoder_layerV
        self.decoder_layerV0 = vae.decoder_layerV0
        
    def forward(self, z, v):
        vd = self.decoder_layerV0(v)
        vd = self.activationv(vd)
        zv = (1.-self.blockZ)*self.decoder_layerZ(z) + (1.-self.blockVdec)*self.decoder_layerV(vd)
        hdd = self.activation(zv)
            
        return torch.exp(self.output_layer(hdd))


#%% VAE with encoder trained using both audio and video. The latent variables are sampled from both "A" and "V"
        
    
class myVAE_AVE(nn.Module):
    
    
    def __init__(self, input_dim=None, latent_dim=None,
                 hidden_dim_encoder=None, activation=None, activationv=None,
                 blockZ = 0., blockVenc = 1., blockVdec = 1., x_block = 0.0, landmarks_dim = 67*67):
        
        super(myVAE_AVE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.landmarks_dim = landmarks_dim 
        self.hidden_dim_encoder = hidden_dim_encoder
        self.activation = activation
        self.activationv = activationv
        self.x_block = x_block
        
        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])       

        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
        self.encoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])
        self.encoder_layerV0 = nn.Linear(self.landmarks_dim, 512)        

        self.blockZ = blockZ
        self.blockVenc = blockVenc
        self.blockVdec = blockVdec
        
        self.output_layer = nn.Linear(hidden_dim_encoder[0], self.input_dim) 
    
        
        #### Define bottleneck layer ####  
        
        self.latent_mean_layer_a = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer_a = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)

        self.latent_mean_layer_v = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer_v = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        
                
    def encodeA(self, x, v):
        xv = self.encoder_layerX(x)
        he = self.activation(xv)
        
        return self.latent_mean_layer_a(he), self.latent_logvar_layer_a(he)

    def encodeV(self, x, v):
        ve = self.encoder_layerV0(v)
        ve = self.activationv(ve)
        xv = self.encoder_layerV(ve)
        he = self.activation(xv)
        
        return self.latent_mean_layer_v(he), self.latent_logvar_layer_v(he)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, v):
        zv = self.decoder_layerZ(z)
        hd = self.activation(zv)
            
        return torch.exp(self.output_layer(hd))

    def forward(self, x, v):
        mu_a, logvar_a = self.encodeA(x, v)
        z_a = self.reparameterize(mu_a, logvar_a)

        mu_v, logvar_v = self.encodeV(x, v)
        z_v = self.reparameterize(mu_v, logvar_v)
        
        return self.decode(z_a, v), mu_a, logvar_a, self.decode(z_v, v), mu_v, logvar_v 
 
        
#%% VAE with encoder trained using both audio and video. The latent variables are sampled from both "A" and "V". 
#   the decoder is also trained conditioned on visual data
        
    
class myVAE_AVEDec(nn.Module):
    
    
    def __init__(self, input_dim=None, latent_dim=None,
                 hidden_dim_encoder=None, activation=None, activationv=None,
                 blockZ = 0., blockVenc = 1., blockVdec = 1., x_block = 0.0, landmarks_dim = 67*67):
        
        super(myVAE_AVEDec, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.landmarks_dim = landmarks_dim 
        self.hidden_dim_encoder = hidden_dim_encoder
        self.activation = activation
        self.activationv = activationv
        self.x_block = x_block
        
        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])       

        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
        self.encoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])
        self.encoder_layerV0 = nn.Linear(self.landmarks_dim, 512)        
        self.decoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])

        self.blockZ = blockZ
        self.blockVenc = blockVenc
        self.blockVdec = blockVdec
        
        self.output_layer = nn.Linear(hidden_dim_encoder[0], self.input_dim) 
    
        
        #### Define bottleneck layer ####  
        
        self.latent_mean_layer_a = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer_a = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)

        self.latent_mean_layer_v = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer_v = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        
                
    def encodeA(self, x, v):
        xv = self.encoder_layerX(x)
        he = self.activation(xv)
        
        return self.latent_mean_layer_a(he), self.latent_logvar_layer_a(he)

    def encodeV(self, x, v):
        ve = self.encoder_layerV0(v)
        ve = self.activationv(ve)
        xv = self.encoder_layerV(ve)
        he = self.activation(xv)
        
        return self.latent_mean_layer_v(he), self.latent_logvar_layer_v(he)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, v):
        ve = self.encoder_layerV0(v)
        ve = self.activationv(ve)
        zv = self.decoder_layerZ(z) + self.decoder_layerV(ve)       
        hd = self.activation(zv)
            
        return torch.exp(self.output_layer(hd))

    def forward(self, x, v):
        mu_a, logvar_a = self.encodeA(x, v)
        z_a = self.reparameterize(mu_a, logvar_a)

        mu_v, logvar_v = self.encodeV(x, v)
        z_v = self.reparameterize(mu_v, logvar_v)
        
        return self.decode(z_a, v), mu_a, logvar_a, self.decode(z_v, v), mu_v, logvar_v



#%% VAE for video data. The input to the encoder is visual data while the output of the decoder reconstructs the corresponding audio data
        
class myVAE_V(nn.Module):
    
    
    def __init__(self, input_dim=None, latent_dim=None,
                 hidden_dim_encoder=None, activation=None, activationv=None,
                 blockZ = 0., blockVenc = 0., blockVdec = 1., x_block = 1.0, landmarks_dim = 67*67):
        
        super(myVAE_V, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.landmarks_dim = landmarks_dim #67*67 #256*5 #100 #67*67 #256*5
        self.hidden_dim_encoder = hidden_dim_encoder
        self.activation = activation
        self.activationv = activationv   
        self.x_block = x_block
        
        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])
        self.decoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])
        self.decoder_layerV0 = nn.Linear(self.landmarks_dim, 512)        

        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
        self.encoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])
        self.encoder_layerV0 = nn.Linear(self.landmarks_dim, 512)        

        self.blockZ = blockZ
        self.blockVenc = blockVenc
        self.blockVdec = blockVdec
        
        self.output_layer = nn.Linear(hidden_dim_encoder[0], self.input_dim) 
    
        
        #### Define bottleneck layer ####  
        
        self.latent_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        
                
    def encode(self, x, v):
        ve = self.encoder_layerV0(v)
        ve = self.activationv(ve)
        xv = (1.-self.x_block)*self.encoder_layerX(x) + (1.-self.blockVenc)*self.encoder_layerV(ve)
        he = self.activation(xv)
        
        return self.latent_mean_layer(he), self.latent_logvar_layer(he)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, v):
        vd = self.decoder_layerV0(v)
        vd = self.activationv(vd)
        zv = (1.-self.blockZ)*self.decoder_layerZ(z) + (1.-self.blockVdec)*self.decoder_layerV(vd)
        hd = self.activation(zv)
            
        return torch.exp(self.output_layer(hd))

    def forward(self, x, v):
        mu, logvar = self.encode(x, v)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z, v), mu, logvar#, self.decode(mu, v)
 
        

#%% Conditional VAE: visual data is considered as a conditioned (deterministic) information
        
class CVAE(nn.Module):
    
    
    def __init__(self, input_dim=None, latent_dim=None,
                 hidden_dim_encoder=None, activation=None, activationV=None):
        
        super(CVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.landmarks_dim = 256*5
        self.hidden_dim_encoder = hidden_dim_encoder
        self.activation = activation
        self.activationV = activationV    

        
        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])
        self.decoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])
        self.decoder_layerV0 = nn.Linear(self.landmarks_dim, 512)        

        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
        self.encoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])
        self.encoder_layerV0 = nn.Linear(self.landmarks_dim, 512)        

        
        # z_prior layer
        self.zprior_layer0 = nn.Linear(self.landmarks_dim, 512)
        self.zprior_layer = nn.Linear(512, self.hidden_dim_encoder[0])
        self.zprior_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.zprior_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)

        
        self.output_layer = nn.Linear(hidden_dim_encoder[0], self.input_dim) 
    
        
        #### Define bottleneck layer ####  
        
        self.latent_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        
                
    def encode(self, x, v):
        ve = self.encoder_layerV0(v)
        ve = self.activationV(ve)
        xv = self.encoder_layerX(x) + self.encoder_layerV(ve)
        he = self.activation(xv)
        
        return self.latent_mean_layer(he), self.latent_logvar_layer(he)

    def zprior(self, v):
        zp0 = self.zprior_layer0(v)
        zp0 = self.activationV(zp0)
        zp = self.zprior_layer(zp0)
        zp = self.activationV(zp)
        
        return self.zprior_mean_layer(zp), self.zprior_logvar_layer(zp)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, v):
        vd = self.decoder_layerV0(v)
        vd = self.activationV(vd)
        zv = self.decoder_layerZ(z) + self.decoder_layerV(vd)
        hd = self.activation(zv)
            
        return torch.exp(self.output_layer(hd))

    def forward(self, x, v):
        mu, logvar = self.encode(x, v)
        z = self.reparameterize(mu, logvar)
        mu_zp, logvar_zp = self.zprior(v)
        z_p = self.reparameterize(mu_zp, logvar_zp)
        
        return self.decode(z, v), mu, logvar, mu_zp, logvar_zp, self.decode(z_p, v)
               
class CDecoder(nn.Module):
    
    def __init__(self, cvae):
        
        super(CDecoder, self).__init__()
        self.latent_dim = cvae.latent_dim
        self.activation = cvae.activation 
        self.activationV = cvae.activationV 
        self.output_layer = None
        self.build(cvae)
        
    def build(self, cvae):
                        
        self.output_layer = cvae.output_layer
        self.decoder_layerZ = cvae.decoder_layerZ
        self.decoder_layerV = cvae.decoder_layerV
        self.decoder_layerV0 = cvae.decoder_layerV0
        
    def forward(self, z, v):
        vd = self.decoder_layerV0(v)
        vd = self.activationV(vd)
        zv = self.decoder_layerZ(z) + self.decoder_layerV(vd)
        hdd = self.activation(zv)
            
        return torch.exp(self.output_layer(hdd))


#%% CVAE with Raw lip images 
        
class CVAER(nn.Module):
    
    
    def __init__(self, input_dim=None, latent_dim=None,
                 hidden_dim_encoder=None, activation=None, activationV=None):
        
        super(CVAER, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.landmarks_dim = 67*67
        self.hidden_dim_encoder = hidden_dim_encoder
        self.activation = activation
        self.activationV = activationV   

        
        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])
        self.decoder_layerV = nn.Linear(128, self.hidden_dim_encoder[0])
        self.decoder_layerV1 = nn.Linear(512, 128)
        self.decoder_layerV0 = nn.Linear(self.landmarks_dim, 512)        

        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
        self.encoder_layerV = nn.Linear(128, self.hidden_dim_encoder[0])
        self.encoder_layerV1 = nn.Linear(512, 128)     
        self.encoder_layerV0 = nn.Linear(self.landmarks_dim, 512)        

        
        # z_prior layer
        self.zprior_layer0 = nn.Linear(self.landmarks_dim, 512)
        self.zprior_layer1 = nn.Linear(512, 128)
        self.zprior_layer = nn.Linear(128, self.hidden_dim_encoder[0])
        self.zprior_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.zprior_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)

        
        self.output_layer = nn.Linear(hidden_dim_encoder[0], self.input_dim) 
    
        
        #### Define bottleneck layer ####  
        
        self.latent_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        
                
    def encode(self, x, v):
        ve0 = self.encoder_layerV0(v)
        ve0 = self.activationV(ve0)
        ve1 = self.encoder_layerV1(ve0)
        ve = self.activationV(ve1)
        xv = self.encoder_layerX(x) + self.encoder_layerV(ve)
        he = self.activation(xv)
        
        return self.latent_mean_layer(he), self.latent_logvar_layer(he)

    def zprior(self, v):
        zp0 = self.zprior_layer0(v)
        zp0 = self.activationV(zp0)
        zp1 = self.zprior_layer1(zp0)
        zp1 = self.activationV(zp1)
        zp = self.zprior_layer(zp1)
        zp = self.activationV(zp)
        
        return self.zprior_mean_layer(zp), self.zprior_logvar_layer(zp)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, v):
        vd0 = self.decoder_layerV0(v)
        vd0 = self.activationV(vd0)
        vd1 = self.decoder_layerV1(vd0)
        vd = self.activationV(vd1)        
        zv = self.decoder_layerZ(z) + self.decoder_layerV(vd)
        hd = self.activation(zv)
            
        return torch.exp(self.output_layer(hd))

    def forward(self, x, v):
        mu, logvar = self.encode(x, v)
        z = self.reparameterize(mu, logvar)
        mu_zp, logvar_zp = self.zprior(v)
        z_p = self.reparameterize(mu_zp, logvar_zp)
        
        return self.decode(z, v), mu, logvar, mu_zp, logvar_zp, self.decode(z_p, v)    
    
    
class CDecoderR(nn.Module):
    
    def __init__(self, cvae):
        
        super(CDecoderR, self).__init__()
        self.latent_dim = cvae.latent_dim
        self.activation = cvae.activation 
        self.activationV = cvae.activationV 
        self.output_layer = None
        self.build(cvae)
        
    def build(self, cvae):
                        
        self.output_layer = cvae.output_layer
        self.decoder_layerZ = cvae.decoder_layerZ
        self.decoder_layerV = cvae.decoder_layerV
        self.decoder_layerV0 = cvae.decoder_layerV0
        self.decoder_layerV1 = cvae.decoder_layerV1
        
    def forward(self, z, v):
        vd0 = self.decoder_layerV0(v)
        vd0 = self.activationV(vd0)
        vd1 = self.decoder_layerV1(vd0)
        vd = self.activationV(vd1) 
        zv = self.decoder_layerZ(z) + self.decoder_layerV(vd)
        hdd = self.activation(zv)
            
        return torch.exp(self.output_layer(hdd))
        

#%% CVAE with Raw lip images and with a feature extractor, shared among all the visual subnetworks, that is learned. The associated network is denoted self.vfeats here.
        
class CVAERTied(nn.Module):
    
    
    def __init__(self, input_dim=None, latent_dim=None,
                 hidden_dim_encoder=None, activation=None, activationV=None):
        
        super(CVAERTied, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.landmarks_dim = 67*67 
        self.hidden_dim_encoder = hidden_dim_encoder
        self.activation = activation
        self.activationV = activationV
           
        self.vfeats = nn.Sequential(
                nn.Linear(self.landmarks_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU() 
                )
        
        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])
        self.decoder_layerV = nn.Linear(128, self.hidden_dim_encoder[0])     

        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
        self.encoder_layerV = nn.Linear(128, self.hidden_dim_encoder[0])       
        
        # z_prior layer
        self.zprior_layer = nn.Linear(128, self.hidden_dim_encoder[0])
        self.zprior_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.zprior_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)

        
        self.output_layer = nn.Linear(hidden_dim_encoder[0], self.input_dim) 
    
        
        #### Define bottleneck layer ####  
        
        self.latent_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        
                
    def encode(self, x, v):
        ve = self.vfeats(v)
        xv = self.encoder_layerX(x) + self.encoder_layerV(ve)
        he = self.activation(xv)
        
        return self.latent_mean_layer(he), self.latent_logvar_layer(he)

    def zprior(self, v):
        zp1 = self.vfeats(v)
        zp = self.zprior_layer(zp1)
        zp = self.activationV(zp)
        
        return self.zprior_mean_layer(zp), self.zprior_logvar_layer(zp)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, v):
        vd = self.vfeats(v)       
        zv = self.decoder_layerZ(z) + self.decoder_layerV(vd)
        hd = self.activation(zv)
            
        return torch.exp(self.output_layer(hd))

    def forward(self, x, v):
        
        mu, logvar = self.encode(x, v)
            
        z = self.reparameterize(mu, logvar)
        
        mu_zp, logvar_zp = self.zprior(v)
        z_p = self.reparameterize(mu_zp, logvar_zp)
        
        return self.decode(z, v), mu, logvar, mu_zp, logvar_zp, self.decode(z_p, v)
         
        
class CDecoderRTied(nn.Module):
    
    def __init__(self, cvae):
        
        super(CDecoderRTied, self).__init__()
        self.latent_dim = cvae.latent_dim
        self.activation = cvae.activation 
        self.activationV = cvae.activationV 
        self.output_layer = None
        self.build(cvae)
        
        
    def build(self, cvae):
        self.vfeats = cvae.vfeats                
        self.output_layer = cvae.output_layer
        self.decoder_layerZ = cvae.decoder_layerZ
        self.decoder_layerV = cvae.decoder_layerV
        
    def forward(self, z, v):
        vd = self.vfeats(v) 
        zv = self.decoder_layerZ(z) + self.decoder_layerV(vd)
        hdd = self.activation(zv)
            
        return torch.exp(self.output_layer(hdd))


