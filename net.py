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
this is the net file, which declares the meta learning network as shown in the paper.
note that the NNs are declared here and the optimization process is implemented in the main file.

@author: F. Zhu and X.Wang
"""
# <editor-fold desc="import package">
import torch
import torch.nn as nn
import numpy as np
from util import *


# </editor-fold>


# <editor-fold desc="meta learning network">

#  customized layer for optimizing the phase shifting matrix
class LambdaLayer(nn.Module):
    def __init__(self, lambda_function):
        super(LambdaLayer, self).__init__()
        self.lambda_function = lambda_function

    def forward(self, x):
        return self.lambda_function(x)


class meta_optimizer_theta(nn.Module):
    """
    this class is used to define the meta learning network for
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        this function is used to initialize the meta learning network for phase shifting matrix
        :param input_size: the size of the input, which is nr_of_RIS_elements in this code
        :param hidden_size: the size of hidden layers, which is hidden_size_theta in this code
        :param output_size: the size of the output, which is nr_of_RIS_elements in this code
        """
        super(meta_optimizer_theta, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
            LambdaLayer(lambda x: 2 * torch.pi * x)
        )

    def forward(self, gradient):
        """
        this function is used to implement the forward propagation of the meta learning network for theta
        :param gradient: the gradient of SE with respect to theta, with sum of user weights normalized to 1
        :return: regulated delta theta
        """
        gradient = gradient.unsqueeze(0)
        gradient = self.layer(gradient)
        gradient = gradient.squeeze(0)
        return gradient


class meta_optimizer_w(nn.Module):
    """
    this class is used to define the meta learning network for w
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        this function is used to initialize the meta learning network for w
        :param input_size: the size of the input, which is nr_of_users*2 in this code
        :param hidden_size: the size of hidden layers, which is hidden_size_w in this code
        :param output_size: the size of the output, which is nr_of_users*2 in this code
        """
        super(meta_optimizer_w, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, gradient):
        """
        this function is used to implement the forward propagation of the meta learning network for w
        :param gradient: the gradient of SE with respect to w, with sum of user weights normalized to 1
        :return: delta w
        """
        gradient = gradient.unsqueeze(0)
        gradient = self.layer(gradient)
        gradient = gradient.squeeze(0)
        return gradient


# </editor-fold>


# <editor-fold desc="build meta-learners">

def meta_learner_w(optimizee, Internal_iteration, user_weights, channel1, channel2, X,
                   theta, noise_power, retain_graph_flag=True):
    """
    Implementation of inner iteration of meta learning for w
    :param optimizee: optimizer for w
    :param Internal_iteration: number of inner loops in each outer loop
    :param user_weights: the weight of each user
    :param channel1: channel G
    :param channel2: channel H
    :param X: the compressed precoding matrix
    :param theta: the phase shift matrix
    :param noise_power: the noise power
    :param retain_graph_flag: whether to retain the graph
    :return: the loss, the accumulated loss, and the updated compressed precoding matrix
    """
    X_internal = X  # initialize the compressed precoding matrix
    X_internal.requires_grad = True  # set the requires_grad flag to true to enable the backward propagation
    sum_loss_w = 0  # record the accumulated loss
    for internal_index in range(Internal_iteration):
        L = -compute_weighted_sum_rate_X(user_weights, channel1, channel2, X_internal, theta, noise_power)
        sum_loss_w = L + sum_loss_w  # accumulate the loss
        L.backward(retain_graph=retain_graph_flag)  # compute the gradient
        X_grad = X_internal.grad.clone().detach()  # clone the gradient
        #  as pytorch can not process complex number, we have to split the real and imaginary parts and concatenate them
        X_grad1 = torch.cat((X_grad.real, X_grad.imag), dim=1)  # concatenate the real and imaginary part
        X_update = optimizee(X_grad1)  # input the gradient and get the increment
        #  recover the complex number from the real and imaginary parts
        X_update1 = X_update[:, 0: nr_of_users] + 1j * X_update[:, nr_of_users: 2 * nr_of_users]
        X_internal = X_internal + X_update1  # update the compressed precoding matrix
        X_update.retain_grad()
        X_internal.retain_grad()
    return L, sum_loss_w, X_internal


def meta_learner_theta(optimizee, Internal_iteration, user_weights, channel1, channel2, X,
                       theta, noise_power, retain_graph_flag=True):
    """
    Implementation of inner iteration of meta learning for theta
    :param optimizee: optimizer for theta
    :param Internal_iteration: number of inner loops in each outer loop
    :param user_weights: the weight of each user
    :param channel1: channel G
    :param channel2: channel H
    :param X: the compressed precoding matrix
    :param theta: the phase shift matrix
    :param noise_power: the noise power
    :param retain_graph_flag: whether to retain the graph
    :return: the loss, the accumulated loss, and the updated phase shift matrix
    """
    cascaded_channel = channel2.conj() @ torch.diag(torch.exp(theta * 1j)) @ channel1
    transmitter_precoder = cascaded_channel.conj().T @ X
    theta_internal = theta
    theta_internal.requires_grad = True
    sum_loss_theta = 0
    for internal_index in range(Internal_iteration):
        L = -compute_weighted_sum_rate(user_weights, channel1, channel2, transmitter_precoder, theta_internal,
                                       noise_power)  # compute the loss
        L.backward(retain_graph=retain_graph_flag)  # compute the gradient
        theta_update = optimizee(theta_internal.grad.clone().detach())  # input the gradient and get the increment
        sum_loss_theta = L + sum_loss_theta  # accumulate the loss
        theta_internal = theta_internal + theta_update  # update the phase shift matrix
        theta_update.retain_grad()
        theta_internal.retain_grad()
    return L, sum_loss_theta, theta_internal


# </editor-fold>

# <editor-fold desc="initialize the network and optimizer">
# initialize the meta learning network w parameters
input_size_w = nr_of_users * 2
hidden_size_w = 200
output_size_w = nr_of_users * 2
batch_size_w = nr_of_users

# initialize the meta learning network theta parameters
input_size_theta = nr_of_RIS_elements
hidden_size_theta = 200
output_size_theta = nr_of_RIS_elements
batch_size_theta = 1

# </editor-fold>


# 测试函数，仅供测试用途
if __name__ == '__main__':
    print("input_size_w: ", input_size_w, "\n",
          "hidden_size_w: ", hidden_size_w, "\n",
          "output_size_w: ", output_size_w, "\n",
          "batch_size_w: ", batch_size_w, "\n",
          "input_size_theta: ", input_size_theta, "\n",
          "hidden_size_theta: ", hidden_size_theta, "\n",
          "output_size_theta: ", output_size_theta, "\n",
          "batch_size_theta: ", batch_size_theta, "\n",
          )
