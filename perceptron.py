# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:20:29 2017

@author: William

Perceptron class.
"""

import numpy as np


class Perceptron(object):
    """A simple perceptron.

    VARIABLES
    - self.input -- the input to the perceptron
    - input_size -- the size of the input
    - self.weights -- the weight matrix of the perceptron
    - weights_size -- the size of the weight matrix

    METHODS
    - add_input -- adds input values
    - initialize_weights -- initialize the weight matrix values
    - update_weights -- update the weight matrix
    - write_weights_to_file -- write the weight matrix into a file
    - read_weights_from_file -- read the weight matrix from a file and
        update the values
    """
    def __init__(self, input_size, weights_size, learning_rate):
        """Allocate arrays for input and weights

        Keyword arguments:
        - input_size -- tuple of input array dimensions
        - weights_size -- tuple of weight matrix array dimensions
        - learning_rate -- float value of learning rate
        """
        self.input = np.zeros(input_size)
        self.weights = np.zeros(weights_size)
        self.learning_rate = learning_rate

    def set_input(self, input_):
        """Add values to input array

        Keyword arguments:
        - input_ -- input array of size input_size
        """
        self.input = input_

    def get_output(self):
        """Produce and return output of perceptron:

        1. Compute the activation potential
        2. Compute sigmoidal output
        """
        pass

    def initialize_weights(self):
        """Initialize the weights randomly"""
        pass

    def update_weights(self):
        """Update the weights with delta rule"""
        pass

    def write_weights_to_file(self, file):
        """Write the weight matrix to a text file

        Keyword arguments:
        - file -- text file to write to
        """
        pass

    def read_weights_from_file(self, file):
        """Read weights from file and update weight matrix

        Keyword arguments:
        - file -- text file to read from
        """
        pass


if __name__ == '__main__':
    pass
