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
    - self.output -- the output of the perceptron
    - input_size -- the size of the input
    - self.weights -- the weight matrix of the perceptron
    - output_size -- the size of the output
    - learning_rate -- the learning rate of the perceptron

    METHODS
    - add_input -- adds input values
    - initialize_weights -- initialize the weight matrix values
    - update_weights -- update the weight matrix
    - write_weights_to_file -- write the weight matrix into a file
    - read_weights_from_file -- read the weight matrix from a file and
        update the values
    """
    def __init__(self, input_size, output_size, learning_rate, linear=False):
        """Allocate arrays for input and weights

        Keyword arguments:
        - input_size -- tuple of input array dimensions
        - output_size -- tuple of weight matrix array dimensions
        - learning_rate -- float value of learning rate
        """
        self.input = np.concatenate((np.zeros(input_size),
                                     np.ones((1, input_size[1]))
                                     )
                                    )
        self.output = np.zeros(output_size)
        self.weights = np.zeros((output_size[0], input_size[0]+1))
        self.learning_rate = learning_rate
        self.linear = linear

    def set_input(self, input_):
        """Add values to input array

        Keyword arguments:
        - input_ -- input array of size input_size
        """
        self.input = np.concatenate((input_, np.ones((1, input_.shape[1]))))

    def get_output(self):
        """Produce and return output of perceptron:

        1. Compute the activation potential
        2. Compute sigmoidal output
        """
        self.out = np.matmul(self.weights, self.input)

        if self.linear:
            self.output = self.out
        else:
            self.output = 1 / (1 + np.exp(-self.out))

        return self.output

    def initialize_weights(self):
        """Initialize the weights randomly"""
        self.weights = np.random.normal(size=self.weights.shape)*0.001

    def update_weights(self, target):
        """Update the weights with delta rule"""
        self.get_output()

        if self.linear:
            _update = (target - self.output) * self.input.T
        else:
            sigmoid_derivative = np.exp(self.out) / (1+np.exp(self.out))**2
            _update = np.matmul(sigmoid_derivative * (target - self.output),
                                self.input.T
                                )

        self.weights += self.learning_rate * _update

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
    """Run"""
    # TESTS
    import matplotlib.pyplot as plt

    input_size = [66*66*3, 1]
    output_size = [66*66*3, 1]
    learning_rate = 0.025
    p = Perceptron(input_size, output_size, learning_rate)
#    p.initialize_weights()
#    p.get_output()
#    p.update_weights(1)

    # GENERATE SAMPLE TRAINING DATA
#    classA = np.zeros([2, 100])
#    classB = np.zeros([2, 100])
#    patterns = np.zeros([2, 200])
#    targets = np.concatenate((np.ones([1, 100]), np.zeros([1, 100])), 1)
#    classA[0, :] = np.random.normal(loc=2.0, size=(1, 100))
#    classA[1, :] = np.random.normal(loc=2.0, size=(1, 100))
#    classB[0, :] = np.random.normal(loc=-2.0, size=(1, 100))
#    classB[1, :] = np.random.normal(loc=-2.0, size=(1, 100))
#    patterns[0, :] = np.concatenate((classA[0, :], classB[0, :]))
#    patterns[1, :] = np.concatenate((classA[1, :], classB[1, :]))
#    permute = np.random.permutation(200)
#    patterns = patterns[:, permute]
#    targets = targets[:, permute]
#
#    plt.plot(patterns[0, np.where(targets >= 1)],
#             patterns[1, np.where(targets >= 1)], '*',
#             patterns[0, np.where(targets <= 0)],
#             patterns[1, np.where(targets <= 0)], '+'
#             )
#
#    learning_rate = 0.001
#    p = Perceptron(patterns.shape, targets.shape, learning_rate)
#    p.set_input(patterns)
#    p.initialize_weights()
#
#    pp = p.weights[0, 0:2]
#    k = -p.weights[0, patterns.shape[0]] / (pp*pp.T)
#    l = np.sqrt(pp*pp.T)
#    plt.clf()
#    plt.xlim(-5, 5)
#    plt.ylim(-5, 5)
#    plt.plot(patterns[0, np.where(targets >= 1)],
#             patterns[1, np.where(targets >= 1)], '*',
#             patterns[0, np.where(targets <= 0)],
#             patterns[1, np.where(targets <= 0)], '+',
#             [pp[0], pp[0]]*k + [-pp[1], pp[1]]/l,
#             [pp[1], pp[1]]*k + [pp[0], -pp[0]]/l, '-'
#             )
#    plt.show()
#    plt.pause(0.1)
#
#    epochs = 20
#    for epoch in range(epochs):
#        p.update_weights(targets)
#
#        pp = p.weights[0, 0:2]
#        k = -p.weights[0, patterns.shape[0]] / (pp*pp.T)
#        l = np.sqrt(pp*pp.T)
#        plt.clf()
#        plt.xlim(-5, 5)
#        plt.ylim(-5, 5)
#        plt.plot(patterns[0, np.where(targets >= 1)],
#                 patterns[1, np.where(targets >= 1)], '*',
#                 patterns[0, np.where(targets <= 0)],
#                 patterns[1, np.where(targets <= 0)], '+',
#                 [pp[0], pp[0]]*k + [-pp[1], pp[1]]/l,
#                 [pp[1], pp[1]]*k + [pp[0], -pp[0]]/l, '-'
#                 )
#        plt.show()
#        plt.pause(0.1)
#        
#        diff = 0.5*abs(targets - p.output)
#        error = np.sum(diff)/patterns.shape[1]
#        print('Error: ', error*100, '%')
