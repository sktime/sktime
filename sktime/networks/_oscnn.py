"""OS-CNN network for time series classification."""

__author__= ["Abhishek Tiwari","SABARNO-PRAMANICK"]

from sktime.networks.base import BaseDeepNetwork
import math
import numpy as np


def calculate_mask_index(kernel_length_now,largest_kernel_lenght):
    """Calculate the indices for zero-padding masks for a given kernel size.

    Parameters
    ----------
    kernel_length_now : int
        Length of the current kernel.
    largest_kernel_length : int
        Length of the largest kernel.

    Returns
    -------
    left_zero_mask_length : int
        Length of the zero padding on the left side.
    right_zero_mask_length : int
        Length of the zero padding on the right side.
    """
    right_zero_mast_length = math.ceil((largest_kernel_lenght-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length+ kernel_length_now

def creat_mask(number_of_input_channel,number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    """Create a mask for constraining kernel weights to a specific size.

    Parameters
    ----------
    number_of_input_channels : int
        Number of input channels for the convolutional layer.
    number_of_output_channels : int
        Number of output channels for the convolutional layer.
    kernel_length_now : int
        Length of the current kernel.
    largest_kernel_length : int
        Length of the largest kernel.

    Returns
    -------
    mask : ndarray
        A mask array to constrain kernel weights.
    """
    ind_left, ind_right= calculate_mask_index(kernel_length_now,largest_kernel_lenght)
    mask = np.ones((number_of_input_channel,number_of_output_channel,largest_kernel_lenght))
    mask[:,:,0:ind_left]=0
    mask[:,:,ind_right:]=0
    return mask


def creak_layer_mask(layer_parameter_list):
    """Generate masks, initial weights, and biases for all layers.

    Parameters
    ----------
    layer_parameter_list : list of tuples
        Each tuple specifies the parameters for a convolutional layer
        (input_channels, output_channels, kernel_length).

    Returns
    -------
    mask : ndarray
        Mask for kernel weights.
    init_weight : ndarray
        Initial weights for the convolutional layers.
    init_bias : ndarray
        Initial biases for the convolutional layers.
    """

    import torch.nn as nn

    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l,ind_r= calculate_mask_index(i[2],largest_kernel_lenght)
        big_weight = np.zeros((i[1],i[0],largest_kernel_lenght))
        big_weight[:,:,ind_l:ind_r]= conv.weight.detach().numpy()

        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)

        mask = creat_mask(i[1],i[0],i[2], largest_kernel_lenght)
        mask_list.append(mask)

    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)


class build_layer_with_layer_parameter():
    """Constructs convolutional layers with kernel masks and initial weights."""

    def __init__(self,layer_parameters):
        """
        Parameters
        ----------
        layer_parameters : list of tuples
            List containing parameters for each convolutional layer.
        """
        super(build_layer_with_layer_parameter, self).__init__()

        import torch
        import torch.nn as nn

        os_mask, init_weight, init_bias= creak_layer_mask(layer_parameters)


        in_channels = os_mask.shape[1] 
        out_channels = os_mask.shape[0] 
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask),requires_grad=False)

        self.padding = nn.ConstantPad1d((int((max_kernel_size-1)/2), int(max_kernel_size/2)), 0)

        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight),requires_grad=True)
        self.conv1d.bias =  nn.Parameter(torch.from_numpy(init_bias),requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        """Forward pass through the layer.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        result : torch.Tensor
            Output tensor after applying the convolutional layer.
        """
        import torch.nn.functional as F

        self.conv1d.weight.data = self.conv1d.weight*self.weight_mask
        #self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        result = F.relu(result_3)
        return result    

class OSCNNNetwork(BaseDeepNetwork):
    """Establish the network structure for OS-CNN.

    Parameters
    ----------
    layer_parameter_list : list of lists
        A list containing the configuration of each layer. Each sublist should contain:
        - `in_channels` (int): Number of input channels.
        - `out_channels` (int): Number of output channels.
        - `kernel_size` (int): Size of the convolutional kernel.
    n_class : int
        Number of output classes for the classification task.
    few_shot : bool, default=True
        If True, excludes the final linear layer for classification.

    Notes
    -----
    The network is constructed with the following components:
    - Oversampling convolutional layers with masks applied to emulate varying kernel sizes.
    - Batch normalization after each convolutional layer.
    - ReLU activation after normalization.
    - Optional fully connected layer if `few_shot` is set to False.

    """
    def __init__(self, layer_parameter_list, n_class, few_shot=True):
        """Initialize the OSCNNNetwork."""

        import torch.nn as nn

        super(OSCNNNetwork, self).__init__()
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []

        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)

        self.net = nn.Sequential(*self.layer_list)
        self.averagepool = nn.AdaptiveAvgPool1d(1)

        out_put_channel_number = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_number += final_layer_parameters[1]

        self.hidden = nn.Linear(out_put_channel_number, n_class)

    def forward(self, X):
        """Define the forward pass of the OSCNNNetwork.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the network.
        """
        import torch

        X = self.net(X)

        X = self.averagepool(X)
        X = X.squeeze_(-1)

        if not self.few_shot:
            X = self.hidden(X)
        return X