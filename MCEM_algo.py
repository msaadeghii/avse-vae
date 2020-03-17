#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Mostafa Sadeghi (mostafa.sadeghi@inria.fr)
License agreement in LICENSE.txt
"""

import numpy as np
import torch

#%% The following implements the MCEM algorithm for audio-only VAE

class MCEM_algo:
    def __init__(self, X=None, W=None, H=None, Z=None, v=None, decoder=None,
                 niter_MCEM=100, niter_MH=40, burnin=30, var_MH=0.01):
        self.X = X # Mixture STFT, shape (F,N)
        self.W = W # NMF dictionary matrix, shape (F, K)
        self.H = H # NMF activation matrix, shape (K, N)
        self.V = self.W @ self.H # Noise variance, shape (F, N)
        self.Z = Z # Last draw of the latent variables, shape (D, N)
        self.v = v
        self.decoder = decoder # VAE decoder, keras model
        self.niter_MCEM = niter_MCEM # Maximum number of MCEM iterations
        self.niter_MH = niter_MH # Number of iterations for the MH algorithm
        # of the E-step
        self.burnin = burnin # Burn-in period for the MH algorithm of the
        # E-step
        self.var_MH = var_MH # Variance of the proposal distribution of the MH
        # algorithm
        self.a = np.ones((1,self.X.shape[1])) # gain parameters, shape (1,N)
        # output of the decoder with self.Z as input, shape (F, N)
        self.Z_mapped_decoder = self.torch2num(self.decoder(self.num2torch(self.Z.T), self.v)).T
        self.speech_var = self.Z_mapped_decoder*self.a # apply gain

    def num2torch(self, x):
        y = torch.from_numpy(x.astype(np.float32))
        return y
    def torch2num(self, x):
        y = x.detach().numpy()
        return y
    
    def metropolis_hastings(self, niter_MH=None, burnin=None):

        if niter_MH==None:
           niter_MH = self.niter_MH

        if burnin==None:
           burnin = self.burnin

        F, N = self.X.shape
        D = self.Z.shape[0]

        Z_sampled = np.zeros((D, N, niter_MH - burnin))

        cpt = 0
        for n in np.arange(niter_MH):

            Z_prime = self.Z + np.sqrt(self.var_MH)*np.random.randn(D,N)

            Z_prime_mapped_decoder = self.torch2num(self.decoder(self.num2torch(Z_prime.T), self.v)).T
            # shape (F, N)
            speech_var_prime = Z_prime_mapped_decoder*self.a # apply gain


            acc_prob = ( np.sum( np.log(self.V + self.speech_var)
                        - np.log(self.V + speech_var_prime)
                        + ( 1/(self.V + self.speech_var)
                        - 1/(self.V + speech_var_prime) )
                        * np.abs(self.X)**2, axis=0)
                        + .5*np.sum( self.Z**2 - Z_prime**2 , axis=0) )

            is_acc = np.log(np.random.rand(1,N)) < acc_prob
            is_acc = is_acc.reshape((is_acc.shape[1],))

            self.Z[:,is_acc] = Z_prime[:,is_acc]
            self.Z_mapped_decoder = self.torch2num(self.decoder(self.num2torch(self.Z.T),self.v)).T
            self.speech_var = self.Z_mapped_decoder*self.a

            if n > burnin - 1:
                Z_sampled[:,:,cpt] = self.Z
                cpt += 1

        return Z_sampled

    def run(self, hop, wlen, win, tol=1e-4):

        F, N = self.X.shape

        X_abs_2 = np.abs(self.X)**2

        cost_after_M_step = np.zeros((self.niter_MCEM, 1))

        for n in np.arange(self.niter_MCEM):

            # MC-Step
            # print('Metropolis-Hastings')
            Z_sampled = self.metropolis_hastings(self.niter_MH, self.burnin)
            Z_sampled_mapped_decoder = np.zeros((F, N, self.niter_MH-self.burnin))
            
            for i in range(self.niter_MH-self.burnin):
                Z_sampled_mapped_decoder[:,:,i] =self.torch2num(self.decoder(self.num2torch(Z_sampled[:,:,i].T),self.v)).T
                    
            speech_var_multi_samples = (Z_sampled_mapped_decoder*
                                        self.a[:,:,None]) # shape (F,N,R)

            # M-Step
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # print('Update W')
            self.W = self.W*(
                    ((X_abs_2*np.sum(V_plus_Z_mapped**-2,
                                     axis=-1)) @ self.H.T)
                    / (np.sum(V_plus_Z_mapped**-1, axis=-1) @ self.H.T)
                    )**.5
            self.V = self.W @ self.H
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # print('Update H')
            self.H = self.H*(
                    (self.W.T @ (X_abs_2 * np.sum(V_plus_Z_mapped**-2,
                                                  axis=-1)))
                    / (self.W.T @ np.sum(V_plus_Z_mapped**-1, axis=-1))
                    )**.5
            self.V = self.W @ self.H
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # print('Update gain')
            self.a = self.a*(
                    (np.sum(X_abs_2 * np.sum(
                            Z_sampled_mapped_decoder*(V_plus_Z_mapped**-2),
                            axis=-1), axis=0) )
                    /(np.sum(np.sum(
                            Z_sampled_mapped_decoder*(V_plus_Z_mapped**-1),
                            axis=-1), axis=0) ) )**.5

            speech_var_multi_samples = (Z_sampled_mapped_decoder*
                                        self.a[:,:,None]) # shape (F,N,R)

            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            cost_after_M_step[n] = np.mean(
                    np.log(V_plus_Z_mapped)
                    + X_abs_2[:,:,None]/V_plus_Z_mapped )

            print("iter %d/%d - cost=%.4f\n" %
                  (n+1, self.niter_MCEM, cost_after_M_step[n]))

            if n>0 and np.abs(cost_after_M_step[n-1] - cost_after_M_step[n]) < tol:
                print('tolerance achieved')
                break

        return cost_after_M_step, n

    def separate(self, niter_MH=None, burnin=None):

        if niter_MH==None:
           niter_MH = self.niter_MH

        if burnin==None:
           burnin = self.burnin

        F, N = self.X.shape

        Z_sampled = self.metropolis_hastings(niter_MH, burnin)
        
        Z_sampled_mapped_decoder = np.zeros((F, N, self.niter_MH-self.burnin))
        
        for i in range(self.niter_MH-self.burnin):
            Z_sampled_mapped_decoder[:,:,i] =self.torch2num(self.decoder(self.num2torch(Z_sampled[:,:,i].T),self.v)).T        

        speech_var_multi_samples = (Z_sampled_mapped_decoder*
                                    self.a[:,:,None]) # shape (F,N,R)

        self.S_hat = np.mean(
                (speech_var_multi_samples/(speech_var_multi_samples
                                           + self.V[:,:,None])),
                                           axis=-1) * self.X

        self.N_hat = np.mean(
                (self.V[:,:,None]/(speech_var_multi_samples
                 + self.V[:,:,None])) , axis=-1) * self.X

#%% The following implements the MCEM algorithm for audio-visual CVAE
        
class MCEM_algo_cvae:
    def __init__(self, mu_z, logvar_z, X=None, W=None, H=None, Z=None, v=None, decoder=None,
                 niter_MCEM=100, niter_MH=40, burnin=30, var_MH=0.01):
        self.X = X # Mixture STFT, shape (F,N)
        self.W = W # NMF dictionary matrix, shape (F, K)
        self.H = H # NMF activation matrix, shape (K, N)
        self.V = self.W @ self.H # Noise variance, shape (F, N)
        self.Z = Z # Last draw of the latent variables, shape (D, N)
        self.v = v
        self.decoder = decoder # VAE decoder, keras model
        self.niter_MCEM = niter_MCEM # Maximum number of MCEM iterations
        self.niter_MH = niter_MH # Number of iterations for the MH algorithm
        # of the E-step
        self.burnin = burnin # Burn-in period for the MH algorithm of the
        # E-step
        self.var_MH = var_MH # Variance of the proposal distribution of the MH
        # algorithm
        self.a = np.ones((1,self.X.shape[1])) # gain parameters, shape (1,N)
        # output of the decoder with self.Z as input, shape (F, N)
        self.Z_mapped_decoder = self.torch2num(self.decoder(self.num2torch(self.Z.T), self.v)).T
        self.speech_var = self.Z_mapped_decoder*self.a # apply gain
        self.muz = mu_z.T
        self.varz = np.exp(logvar_z).T
        
    def num2torch(self, x):
        y = torch.from_numpy(x.astype(np.float32))
        return y
    def torch2num(self, x):
        y = x.numpy()
        return y
    
    def metropolis_hastings(self, niter_MH=None, burnin=None):

        if niter_MH==None:
           niter_MH = self.niter_MH

        if burnin==None:
           burnin = self.burnin

        F, N = self.X.shape
        D = self.Z.shape[0]

        Z_sampled = np.zeros((D, N, niter_MH - burnin))

        cpt = 0
        for n in np.arange(niter_MH):

            Z_prime = self.Z + np.sqrt(self.var_MH)*np.random.randn(D,N)

            Z_prime_mapped_decoder = self.torch2num(self.decoder(self.num2torch(Z_prime.T), self.v)).T
#            
            # shape (F, N)
            speech_var_prime = Z_prime_mapped_decoder*self.a # apply gain

            acc_prob = ( np.sum( np.log(self.V + self.speech_var)
                        - np.log(self.V + speech_var_prime)
                        + ( 1/(self.V + self.speech_var)
                        - 1/(self.V + speech_var_prime) )
                        * np.abs(self.X)**2, axis=0)
                        + .5*np.sum( ((self.Z-self.muz)**2 - (Z_prime-self.muz)**2)/self.varz , axis=0) )

            is_acc = np.log(np.random.rand(1,N)) < acc_prob
            is_acc = is_acc.reshape((is_acc.shape[1],))

            self.Z[:,is_acc] = Z_prime[:,is_acc]
            self.Z_mapped_decoder = self.torch2num(self.decoder(self.num2torch(self.Z.T),self.v)).T
            self.speech_var = self.Z_mapped_decoder*self.a

            if n > burnin - 1:
                Z_sampled[:,:,cpt] = self.Z
                cpt += 1

        return Z_sampled

    def run(self, hop, wlen, win, tol=1e-4):

        F, N = self.X.shape

        X_abs_2 = np.abs(self.X)**2

        cost_after_M_step = np.zeros((self.niter_MCEM, 1))

        for n in np.arange(self.niter_MCEM):

            # MC-Step
            # print('Metropolis-Hastings')
            Z_sampled = self.metropolis_hastings(self.niter_MH, self.burnin)
            Z_sampled_mapped_decoder = np.zeros((F, N, self.niter_MH-self.burnin))
            
            for i in range(self.niter_MH-self.burnin):
                Z_sampled_mapped_decoder[:,:,i] =self.torch2num(self.decoder(self.num2torch(Z_sampled[:,:,i].T),self.v)).T
                    

            speech_var_multi_samples = (Z_sampled_mapped_decoder*
                                        self.a[:,:,None]) # shape (F,N,R)

            # M-Step
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # print('Update W')
            self.W = self.W*(
                    ((X_abs_2*np.sum(V_plus_Z_mapped**-2,
                                     axis=-1)) @ self.H.T)
                    / (np.sum(V_plus_Z_mapped**-1, axis=-1) @ self.H.T)
                    )**.5
            self.V = self.W @ self.H
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # print('Update H')
            self.H = self.H*(
                    (self.W.T @ (X_abs_2 * np.sum(V_plus_Z_mapped**-2,
                                                  axis=-1)))
                    / (self.W.T @ np.sum(V_plus_Z_mapped**-1, axis=-1))
                    )**.5
            self.V = self.W @ self.H
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # print('Update gain')
            self.a = self.a*(
                    (np.sum(X_abs_2 * np.sum(
                            Z_sampled_mapped_decoder*(V_plus_Z_mapped**-2),
                            axis=-1), axis=0) )
                    /(np.sum(np.sum(
                            Z_sampled_mapped_decoder*(V_plus_Z_mapped**-1),
                            axis=-1), axis=0) ) )**.5

            speech_var_multi_samples = (Z_sampled_mapped_decoder*
                                        self.a[:,:,None]) # shape (F,N,R)

            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            cost_after_M_step[n] = np.mean(
                    np.log(V_plus_Z_mapped)
                    + X_abs_2[:,:,None]/V_plus_Z_mapped )

            print("iter %d/%d - cost=%.4f\n" %
                  (n+1, self.niter_MCEM, cost_after_M_step[n]))

            if n>0 and np.abs(cost_after_M_step[n-1] - cost_after_M_step[n]) < tol:
                print('tolerance achieved')
                break

        return cost_after_M_step, n

    def separate(self, niter_MH=None, burnin=None):

        if niter_MH==None:
           niter_MH = self.niter_MH

        if burnin==None:
           burnin = self.burnin

        F, N = self.X.shape

        Z_sampled = self.metropolis_hastings(niter_MH, burnin)

        Z_sampled_mapped_decoder = np.zeros((F, N, self.niter_MH-self.burnin))
        
        for i in range(self.niter_MH-self.burnin):
            Z_sampled_mapped_decoder[:,:,i] =self.torch2num(self.decoder(self.num2torch(Z_sampled[:,:,i].T),self.v)).T        

        speech_var_multi_samples = (Z_sampled_mapped_decoder*
                                    self.a[:,:,None]) # shape (F,N,R)

        self.S_hat = np.mean(
                (speech_var_multi_samples/(speech_var_multi_samples
                                           + self.V[:,:,None])),
                                           axis=-1) * self.X

        self.N_hat = np.mean(
                (self.V[:,:,None]/(speech_var_multi_samples
                 + self.V[:,:,None])) , axis=-1) * self.X


#%% The following implements the MCEM algorithm for audio-visual CVAE
        
class VI_MCEM_algo:
    def __init__(self, mu_z, logvar_z, X=None, W=None, H=None, Z=None, v=None, decoder=None,
                 niter_VI_MCEM=100, niter_MH=40, burnin=30, var_MH=0.01):
        self.X = X # Mixture STFT, shape (F,N)
        self.W = W # NMF dictionary matrix, shape (F, K)
        self.H = H # NMF activation matrix, shape (K, N)
        self.V = self.W @ self.H # Noise variance, shape (F, N)
        self.Z = Z # Last draw of the latent variables, shape (D, N)
        self.v = v
        self.decoder = decoder # VAE decoder, keras model
        self.niter_VI_MCEM = niter_VI_MCEM # Maximum number of MCEM iterations
        self.niter_MH = niter_MH # Number of iterations for the MH algorithm
        # of the E-step
        self.burnin = burnin # Burn-in period for the MH algorithm of the
        # E-step
        self.var_MH = var_MH # Variance of the proposal distribution of the MH
        # algorithm
        self.a = np.ones((1,self.X.shape[1])) # gain parameters, shape (1,N)
        # output of the decoder with self.Z as input, shape (F, N)
        self.Z_mapped_decoder = self.torch2num(self.decoder(self.num2torch(self.Z.T), self.v)).T
        self.speech_var = self.Z_mapped_decoder*self.a # apply gain
        self.muz = mu_z.T
        self.varz = np.exp(logvar_z).T
        self.eps = np.finfo(float).eps
        
    def num2torch(self, x):
        y = torch.from_numpy(x.astype(np.float32))
        return y
    def torch2num(self, x):
        y = x.numpy()
        return y
    
    def metropolis_hastings_AVz(self, abs_Sp2, muz = None, varz = None, niter_MH=40, burnin=30, var_MH = 0.01):

        N = self.X.shape[1]
        D = self.Z.shape[0]

        Z_sampled = np.zeros((D, N, niter_MH - burnin))
        
        if muz == None:
            muz = self.muz
            
        if varz == None:
            varz = self.varz
            
        cpt = 0
        
        for n in np.arange(niter_MH):

            Z_prime = self.Z + np.sqrt(self.var_MH)*np.random.randn(D,N)
            with torch.no_grad():
                Z_prime_mapped_decoder = self.torch2num(self.decoder(self.num2torch(Z_prime.T), self.v)).T
            # shape (F, N)
            speech_var_prime = Z_prime_mapped_decoder

            acc_prob = ( np.sum( np.log(self.speech_var)
                        - np.log(speech_var_prime)
                        + ( 1/(self.speech_var)
                        - 1/(speech_var_prime) )
                        * abs_Sp2, axis=0)
                        + .5*np.sum( ((self.Z-muz)**2 - (Z_prime-muz)**2)/varz , axis=0) )

            is_acc = np.log(np.random.rand(1,N)) < acc_prob
            is_acc = is_acc.reshape((is_acc.shape[1],))

            self.Z[:,is_acc] = Z_prime[:,is_acc]
            with torch.no_grad():
                self.Z_mapped_decoder = self.torch2num(self.decoder(self.num2torch(self.Z.T),self.v)).T
            self.speech_var = self.Z_mapped_decoder

            if n > burnin - 1:
                Z_sampled[:,:,cpt] = self.Z
                cpt += 1

        return Z_sampled


    def run(self, tol=1e-4):

        F, N = self.X.shape
        
        self.Mu_s = self.X.copy()
        self.Sig_s = np.zeros((F, N)) 
        
        for n in np.arange(self.niter_VI_MCEM):

            ############ E-s step ############
            with torch.no_grad():
                z_samples = np.transpose(self.metropolis_hastings_AVz(np.abs(self.Mu_s)**2+self.Sig_s, muz = None,
                                                                  varz = None, niter_MH=40, burnin=30, var_MH = 0.01),(1,2,0)) # shape (N, NumSamples, 32)

                Z_decoded = np.transpose(self.torch2num(self.decoder(self.num2torch(z_samples), self.v[:,None,:])),(2,1,0))     # shape (F, N, NumSamples)          


            inv_gam = np.mean(1./Z_decoded, axis = 1)
            gam = 1./inv_gam
            
            self.Sig_s = (self.V*gam)/(self.V + gam) 
            self.Mu_s = (gam/(self.V + gam))*self.X 
              
            Vm = np.abs(self.X-self.Mu_s)**2+self.Sig_s
            
            ############ Update W ############
        
            self.W = self.W*( (((self.V**-2) * Vm) @ self.H.T)
                    / ((self.V**-1) @ self.H.T) )
            
            self.W = np.maximum(self.W, self.eps)
            
            self.V = self.W @ self.H
            
            ############ Update H ############
            
            self.H = self.H*( (self.W.T @ ( (self.V**-2) * Vm )) 
                    / (self.W.T @ (self.V**-1)))
   
            self.H = np.maximum(self.H, self.eps)
            
            norm_col_W = np.sum(np.abs(self.W), axis=0)
            self.W = self.W/norm_col_W[np.newaxis,:]
            self.H = self.H*norm_col_W[:,np.newaxis]
               
            self.V = self.W @ self.H

            print("iter %d/%d - cost=%.4f\n" %
                  (n+1, self.niter_VI_MCEM, 0))
            
        # Estimate clean speech:
        S_hat = (gam/(self.V + gam))*self.X
        N_hat = self.X - S_hat
        
        return S_hat, N_hat

    def separate(self, niter_MH=None, burnin=None):

        if niter_MH==None:
           niter_MH = self.niter_MH

        if burnin==None:
           burnin = self.burnin

        F, N = self.X.shape

        with torch.no_grad():
            z_samples = np.transpose(self.metropolis_hastings_AVz(np.abs(self.Mu_s)**2+self.Sig_s, muz = None,
                                                              varz = None, niter_MH=40, burnin=30, var_MH = 0.01),(1,2,0)) # shape (N, NumSamples, 32)

            Z_decoded = np.transpose(self.torch2num(self.decoder(self.num2torch(z_samples), self.v[:,None,:])),(2,1,0))     # shape (F, N, NumSamples)          


        inv_gam = np.mean(1./Z_decoded, axis = 1)
        gam = 1./inv_gam
          
        # Estimate clean speech:
        self.S_hat = (gam/(self.V + gam))*self.X
        self.N_hat = self.X - self.S_hat
        
        return None

