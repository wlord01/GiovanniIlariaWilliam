# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:15:38 2017

@author: William

Functions:
- plot - plots the ignorance and perceptron outpot of the objects
"""


import matplotlib.pyplot as plt
import numpy as np


def plot(file_name):
    """Plot data

    Keyword arguents:
    - file_name -- name of data file (string)

    Reads data from file and plots ignorance and perceptron output for
    all objects in the phase 1 simulation.
    """
    data_array = np.load(file_name)

    # PLOT IGNORANCE FOR ALL OBJECTS
    plt.figure()
    plt.title('Ignorance')
    for i in range(len(data_array[0, 0, :])):
        plt.plot(np.arange(data_array.shape[0]), data_array[:, 0, i],
                 label='Object '+str(i+1)
                 )
    plt.legend()

    # PLOT PERCEPTRON OUTPUT FOR ALL OBJECTS
    plt.figure()
    plt.title('Perceptron output')
    for i in range(len(data_array[0, 0, :])):
        plt.plot(np.arange(data_array.shape[0]), data_array[:, 1, i],
                 label='Object '+str(i+1)
                 )
    plt.legend()


if __name__ == '__main__':
    """Main"""
    # TESTS
