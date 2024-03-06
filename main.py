"""
GEMML code
------------------------------
Implementation of GEMML algorithm, which is proposed in the paper:
Robust Beamforming for RIS-aided Communications: Gradient Enhanced Manifold Meta Learning

References and Relevant Links
------------------------------
GitHub Repository:
https://github.com/FenghaoZhu/GEMML

Related arXiv Paper:
https://arxiv.org/abs/xxxx.xxxxx

file introduction
------------------------------
this is the main function which can be run directly

@author: F. Zhu and X.Wang
"""
# <editor-fold desc="import package">
import random
import scipy.io as sio
import torch

from net import *
from tqdm import tqdm
import math

# </editor-fold>


# <editor-fold desc="set random seed">
seed = 42  # fix the random seed
torch.manual_seed(seed)  # cpu random seed
torch.cuda.manual_seed(seed)  # gpu random seed
torch.cuda.manual_seed_all(seed)  # multi-gpu random seed
np.random.seed(seed)  # numpy random seed
random.seed(seed)  # python random seed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# </editor-fold>

# <editor-fold desc="load channel">
H_t = sio.loadmat(f'dataset.mat')['HH']  # load the channel H, numpy format
G_t = sio.loadmat(f'dataset.mat')['GG']  # load the channel G, numpy format
user_weights = sio.loadmat(f'dataset.mat')['omega'].squeeze()  # load the user weights, numpy format
regulated_user_weights = user_weights / np.sum(user_weights)  # normalize the user weights
H_t = torch.tensor(H_t)  # transforms from numpy to torch format
G_t = torch.tensor(G_t)  # transforms from numpy to torch format
# </editor-fold>

# <editor-fold desc="training process">
WSR_list_per_sample = torch.zeros(nr_of_training, External_iteration)  # record the WSR of each sample
# Iterate and optimize each sample
for item_index in range(nr_of_training):
    # refresh the nn parameters at the beginning of each sample to guarantee the independence
    # note that GEMML is pretraining free!

    # initialize the meta learning network for the precoding matrix
    optimizer_w = meta_optimizer_w(input_size_w, hidden_size_w, output_size_w)
    # initialize the optimizer for the precoding matrix
    adam_w = torch.optim.Adam(optimizer_w.parameters(), lr=optimizer_lr_w)

    # initialize the meta learning network for the phase shift matrix
    optimizer_theta = meta_optimizer_theta(input_size_theta, hidden_size_theta, output_size_theta)
    # initialize the optimizer for the phase shift matrix
    adam_theta = torch.optim.Adam(optimizer_theta.parameters(), lr=optimizer_lr_theta)

    maxi = 0  # record the maximum WSR of each sample
    # load the channel sample
    G = G_t[:, :, item_index].to(torch.complex64)  # dimension: nr_of_RIS_elements * nr_of_BS_antennas
    H = H_t[:, :, item_index].to(torch.complex64)  # dimension: nr_of_users * nr_of_RIS_elements

    #  initialize the precoding matrix and the phase shift matrix
    theta = torch.randn(nr_of_RIS_elements).to(torch.float32)  # initialize the phase shift matrix
    theta_init = theta
    cascaded_channel = H.conj() @ torch.diag(torch.exp(theta * 1j)) @ G  # cascaded channel

    #  initialize the precoding matrix and the compressed precoding matrix
    X, V = init_X(nr_of_BS_antennas, nr_of_users, cascaded_channel, total_power)
    X_init = X
    transmitter_precoder_init = V
    transmitter_precoder = transmitter_precoder_init

    LossAccumulated_w = 0  # record the accumulated loss in the meta learning network for precoding matrix
    LossAccumulated_theta = 0  # record the accumulated loss in the meta learning network for phase shift matrix
    for epoch in range(External_iteration):
        #  update the precoding matrix and the phase shift matrix in outer loop
        #  one outer loop includes Internal_iteration inner loops
        #  when updating the phase shift matrix, the compressed precoding matrix is inherited from the last outer loop
        loss_theta, sum_loss_theta, theta = meta_learner_theta(optimizer_theta, Internal_iteration,
                                                               regulated_user_weights, G, H,
                                                               X.clone().detach(),  # clone the precoding matrix
                                                               theta_init,  # update the phase shift matrix from scratch
                                                               noise_power)
        #  when updating the compressed precoding matrix, the phase shift matrix is inherited from the last outer loop
        loss_w, sum_loss_w, X = meta_learner_w(optimizer_w, Internal_iteration,
                                               regulated_user_weights, G, H,
                                               X_init,  # update the precoding matrix from scratch
                                               theta.clone().detach(),  # clone the phase shift matrix
                                               noise_power)

        # handle the normalization of the compressed precoding matrix
        cascaded_channel = H.conj() @ torch.diag(torch.exp(theta * 1j)) @ G  # cascaded channel
        transmitter_precoder = cascaded_channel.conj().T @ X  # compute the precoding matrix before normalization
        normV = torch.norm(transmitter_precoder)  # compute the norm of the precoding matrix before normalization
        WW = math.sqrt(total_power) / normV  # normalization coefficient
        X = X * WW  # normalize the compressed precoding matrix
        transmitter_precoder = transmitter_precoder * WW  # normalize the precoding matrix

        # compute the loss of each sample
        loss_total = -compute_weighted_sum_rate(regulated_user_weights, G, H, transmitter_precoder, theta, noise_power)
        LossAccumulated_w = LossAccumulated_w + loss_total  # accumulate the precoding matrix network loss
        LossAccumulated_theta = LossAccumulated_theta + loss_total  # accumulate the shift matrix network loss
        MSR = compute_weighted_sum_rate(user_weights, G, H, transmitter_precoder, theta.detach(), noise_power)
        WSR_list_per_sample[item_index, epoch] = MSR  # record the WSR of each sample
        if MSR > maxi:  # update maxi only when the WSR is larger than the current maximum WSR
            maxi = MSR.item()  # record the maximum WSR of each sample
            print('max', maxi, 'epoch=', epoch, 'item', item_index)  # print the maximum WSR of each sample
        if (epoch + 1) % Update_steps == 0:  # update the meta learning network every Update_steps outer loops
            adam_w.zero_grad()
            adam_theta.zero_grad()
            Average_loss_w = LossAccumulated_w / Update_steps
            Average_loss_theta = LossAccumulated_theta / Update_steps
            Average_loss_w.backward(retain_graph=True)
            Average_loss_theta.backward(retain_graph=True)
            adam_w.step()
            if (epoch + 1) % 5 == 0:
                adam_theta.step()
            MSR = compute_weighted_sum_rate(regulated_user_weights, G, H, transmitter_precoder, theta.detach(),
                                            noise_power)
            LossAccumulated_w = 0  # reset the accumulated loss in the meta learning network for precoding matrix
            LossAccumulated_theta = 0  # reset the accumulated loss in the meta learning network for phase shift matrix

    #  save the WSR of each sample
    WSR_matrix = WSR_list_per_sample
    sio.savemat(f'./GEMML_result.mat',
                {'WSR_matrix': WSR_matrix.detach().numpy()})

# </editor-fold>
