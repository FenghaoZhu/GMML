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
this is the utils file, including the initialization of the channel, the computation of the SINR and the rate, etc.

@author: F. Zhu and X.Wang
"""
# <editor-fold desc="import package">
import numpy as np
import torch
import torch.nn as nn
import random
# </editor-fold>

# <editor-fold desc="define the constant">
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
External_iteration = 500

Internal_iteration = 1
Update_steps = 1
N_i = Internal_iteration
N_o = Update_steps
# optimizer_lr_theta = 10e-4  # changeable
# optimizer_lr_w = 15e-4
optimizer_lr_theta = 1e-3  # changeable
optimizer_lr_w = 1.5e-3
hidden_size_theta = 200
hidden_size_w = 200

nr_of_users = 4
nr_of_BS_antennas = 64
nr_of_RIS_elements = 100

epoch = 1
nr_of_training = 100 # used for training, while solving
scheduled_users = [x for x in range(nr_of_users)]
selected_users = [x for x in range(nr_of_users)]  # array of scheduled users. Note that we schedule all the users.
snr = 10
noise_power = 1
total_power = noise_power * 10 ** (snr / 10)

# </editor-fold>

# <editor-fold desc="define the util function">
def initialize_channel(number_of_BS_antennas, number_of_users):
    """
    Generate the channel matrix
    :param number_of_BS_antennas: the number of BS antennas
    :param number_of_users: the number of users
    :return: channel matrix
    """
    channel = torch.randn(number_of_users, number_of_BS_antennas) + 1j * torch.randn(number_of_users,
                                                                                     number_of_BS_antennas)
    channel = channel / torch.sqrt(torch.tensor(2))
    return channel  # size: nr_of_users * nr_of_BS_antennas


def compute_sinr(channel1, channel2, precoder, theta, power_noise, user_id):
    """
    This version of SINR computation deals with torch format, precoder is a complex matrix
    :param channel1: nr_of_RIS_elements * nr_of_BS_antennas  (G in our paper, the channel from BS to RIS)
    :param channel2: nr_of_users * nr_of_RIS_elements   (h in our paper, the channel from RIS to users)
    :param precoder: nr_of_BS_antennas * nr_of_users  (w in our paper, the precoder of BS)
    :param theta: nr_of_RIS_elements * nr_of_RIS_elements (theta in our paper, the phase shift of RIS)
    :param user_id: the index of the user
    :param power_noise: the noise power
    :return: the SINR of the user
    """
    # to avoid duplicate calculation, we first calculate G @ w_k and h_k @ Theta
    # and then calculate the numerator and denominator using the result
    htG = torch.conj(channel2[user_id, :])*torch.exp(theta * 1j) @ channel1
    inter_user_interference = (torch.absolute(htG @ precoder)) ** 2
    numerator = inter_user_interference[user_id]
    inter_user_interference = torch.sum(inter_user_interference)-numerator
    denominator = power_noise + inter_user_interference
    result = numerator / denominator
    return result


def compute_weighted_sum_rate(user_weights, channel1, channel2, precoder_in, theta, power_noise):
    """
    This version of rate function deals with torch format, and the transmitter. Precoder is a complex matrix
    :param user_weights: the weights of users
    :param channel1: nr_of_RIS_elements * nr_of_BS_antennas  (G in our paper, the channel from BS to RIS)
    :param channel2: nr_of_users * nr_of_RIS_elements   (h in our paper, the channel from RIS to users)
    :param precoder_in: nr_of_BS_antennas * nr_of_users  (w in our paper, the precoder of BS)
    :param theta: nr_of_RIS_elements * 1 (theta in our paper, the phase shift of RIS)
    :param power_noise: the noise power
    :return: weighted_sum_rate  (the weighted sum rate of the users)
    """
    result = 0
    nr_of_user = channel2.shape[0]
    transmitter_precoder = precoder_in
    for user_index in range(nr_of_user):
        user_sinr = compute_sinr(channel1, channel2, transmitter_precoder, theta, power_noise, user_index)
        result = result + user_weights[user_index] * torch.log2(1 + user_sinr)
    # print('end')
    return result


def compute_weighted_sum_rate_X(user_weights, channel1, channel2, X, theta, power_noise):
    """
    This version of rate function deals with torch format, and the transmitter. Precoder is a complex matrix
    :param user_weights: the weights of users
    :param channel1: nr_of_RIS_elements * nr_of_BS_antennas  (G in our paper, the channel from BS to RIS)
    :param channel2: nr_of_users * nr_of_RIS_elements   (h in our paper, the channel from RIS to users)
    :param X: nr_of_users * nr_of_users  (X in our paper, the collapsed precoder of BS)
    :param theta: nr_of_RIS_elements * 1 (theta in our paper, the phase shift of RIS)
    :param power_noise: the noise power
    :return: weighted_sum_rate  (the weighted sum rate of the users)
    """
    result = 0
    cascaded_channel = channel2.conj() @ torch.diag(torch.exp(theta * 1j)) @ channel1
    transmitter_precoder = cascaded_channel.conj().T @ X
    nr_of_user = channel2.shape[0]
    for user_index in range(nr_of_user):
        user_sinr = compute_sinr(channel1, channel2, transmitter_precoder, theta, power_noise, user_index)
        result = result + user_weights[user_index] * torch.log2(1 + user_sinr)
    return result


def init_transmitter_precoder(channel_realization):
    """
    This function is used to initialize the transmitter precoder in torch and numpy format
    :param channel_realization:
    :return: transmitter_precoder, transmitter_precoder_initialize
    """
    transmitter_precoder_init = np.zeros((nr_of_BS_antennas, nr_of_users)) + 1j * np.zeros(
        (nr_of_BS_antennas, nr_of_users))
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            transmitter_precoder_init[:, user_index] = channel_realization[user_index, :]
    transmitter_precoder_initialize = transmitter_precoder_init / np.linalg.norm(transmitter_precoder_init) * np.sqrt(
        total_power)

    transmitter_precoder_init = torch.from_numpy(transmitter_precoder_initialize)
    transmitter_precoder_complex = transmitter_precoder_init
    transmitter_precoder_Re = transmitter_precoder_complex.real
    transmitter_precoder_Im = transmitter_precoder_complex.imag
    transmitter_precoder = torch.cat((transmitter_precoder_Re, transmitter_precoder_Im), dim=1)
    return transmitter_precoder, transmitter_precoder_initialize  # torch real format, numpy complex format


def init_X(antenna_number, user_number, cascaded_channel, power):
    """
    This function is used to initialize the collapsed beamforming vector in torch format
    :param user_number: the number of users
    :param cascaded_channel: the cascaded channel between BS and users
    :param power: the power constraint
    :return: initiliazed collapsed beamforming vector X (user_number * user_number)
     and initiliazed full beamforming vector V (antenna_number * user_number)
    """
    # initialize the collapsed beamforming vector X and the full beamforming vector V
    X = torch.randn(user_number, user_number) + 1j * torch.randn(user_number, user_number)
    V = torch.randn(antenna_number, user_number) + 1j * torch.randn(antenna_number, user_number)
    V = cascaded_channel.conj().T @ X
    # normalize the collapsed beamforming vector X and the full beamforming vector V
    X = X / torch.norm(V) * torch.sqrt(torch.tensor(power))
    V = cascaded_channel.conj().T @ X
    return X, V


# </editor-fold>


# 测试函数，仅供测试用途  Test function, for test only
if __name__ == '__main__':
    channel1 = torch.randn(nr_of_RIS_elements, nr_of_BS_antennas) + 1j * torch.randn(
        nr_of_RIS_elements, nr_of_BS_antennas)
    channel2 = torch.randn(nr_of_users, nr_of_RIS_elements) + 1j * torch.randn(
        nr_of_users, nr_of_RIS_elements)
    precoder = torch.randn(nr_of_BS_antennas, nr_of_users) + 1j * torch.randn(
        nr_of_BS_antennas, nr_of_users)
    theta = torch.randn(nr_of_RIS_elements)
    user_weights = np.ones(nr_of_users)
    noise_power = 1
    user_id = 0
    selected_users = [x for x in range(nr_of_users)]
    print("compute_weighted_sum_rate: ",
          compute_weighted_sum_rate(user_weights, channel1, channel2,
                                    precoder, theta, noise_power),
          '\n',
          "compute_sinr: ",
          compute_sinr(channel1, channel2, precoder, theta, noise_power, user_id),
          )
