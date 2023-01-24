# -*- coding: utf-8 -*-
"""Time Convolutional Neural Network (CNN) (minus the final output layer)."""

__author__ = "James Large, Withington, Tony Bagnall"

from tensorflow import keras

from sktime.networks.base import BaseDeepNetwork

import keras
# from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN

class RNNNetwork(BaseDeepNetwork):
    """
    Adapter from the implementation from 
    
    """
    def __init__(
        self,
        units=50,
        random_state = 0
    ):
        self.random_state = random_state
        self.units = units
        
    def build_network(self, input_shape, **kwargs):
        input_layer = keras.layers.Input((input_shape,1))
        
        rnn = SimpleRNN(units = self.units,return_sequences= True,input_shape = (input_shape,1))(input_layer)
        dense = Dense(units = 1)(rnn)
        
        return input_layer,dense 
        
        
    
    