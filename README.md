## AV-VAE Speech Enhancement

This repository contains PyTorch implementations of the audio-visual speech enhancement methods based on variational autoencoder (VAE), presented in [1] and [2].

The VAE architectures are defined in `AV_VAE.py`.

To train the VAE model using clean audio and video data, use `train_VAE.py`.

`TCD_TIMIT.py` contains a custom dataset loader for the TCD-TIMIT dataset.

The MCEM algorithm for speech enhancement is impelemented in `MCEM_algo.py`.

To enhacne a given speech, use `speech_enhancer_VAE.py`.

## References:

[1] M. Sadeghi,  S. Leglaive, X. Alameda-Pineda, L. Girin, and R. Horaud, “Audio-visual Speech Enhancement Using Conditional Variational Auto-Encoder,” IEEE Transactions on Audio, Speech and Language Processing, vol. 28, pp. 1788- 1800, May 2020.

[2] M. Sadeghi and X. Alameda-Pineda, “Robust Unsupervised Audio-visual Speech Enhancement Using a Mixture of Variational Autoencoders,” in IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP), Barcelona, Spain, May 2020.
