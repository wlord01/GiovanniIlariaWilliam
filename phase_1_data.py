# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:15:38 2017

@author: William

Functions:
- get_average_data(number_of_simulations) - Averages the data over the
  number of simulations defined by the function argument
  number_of_simulations.
- plot(file_name) - plots the ignorance and predictor outputs for the
  objects.
"""


import matplotlib.pyplot as plt
import numpy as np


def get_average_data(number_of_simulations):
    """Calculate average data over many simulations

    Keyword_arguments:
    - number_of_simulations -- Number of simulations (int) over which
      to calculate average data

    Concatenates the data from all the simulations and calculates
    average values for ignorance and predictor outputs at each time
    step over the simulations. A data file is created and saved with
    the average data.
    """
    all_data = np.load('s1data_array.npy')

    for i in range(2, number_of_simulations + 1):
        file_name = 's{}data_array.npy'.format(str(i))
        simulation_data = np.load(file_name)
        all_data[:, 2:5] += simulation_data[:, 2:5]

    average_data = all_data
    average_data[:, 2:5] = average_data[:, 2:5] / number_of_simulations
    average_data_file_name = '{}sim_average_data.npy'.format(
        str(number_of_simulations)
        )
    np.save(average_data_file_name, average_data)


def plot(file_name):
    """Plot data

    Keyword arguents:
    - file_name -- name of data file (string)

    Reads data from file and plots ignorance and perceptron output for
    all objects in the phase 1 simulation.
    """
    def get_details(data_array, i, j):
        """Get details for plots

        Keyword arguments:
        - data_array -- array of data from simulation
        - i -- action number (integer)
        - j -- object number (integer)

        Takes information from data array. Returns strings for line color,
        marker, label and points for markers.
        """
        if data_array[0, 0, j, i] == 0:
            object_type = 's'
            label = 'Square, '
        elif data_array[0, 0, j, i] == 1:
            object_type = 'o'
            label = 'Circle, '
        elif data_array[0, 0, j, i] == 2:
            object_type = '*'
            label = 'Rectangle, '

        if data_array[0, 1, j, i] == 0:
            object_color = 'r'
            label += 'red'
        elif data_array[0, 1, j, i] == 1:
            object_color = 'g'
            label += 'green'
        elif data_array[0, 1, j, i] == 2:
            object_color = 'b'
            label += 'blue'

        line_color = object_color
        marker = object_color+object_type
        marker_points = np.arange(0, len(data_array),
                                  int(len(data_array)/3)
                                  )
        return line_color, marker, marker_points, label

    data_array = np.load(file_name)

    # PLOT DATA FOR ALL ACTIONS
    for i in range(len(data_array[0, 0, 0, :])):
        # PLOT IGNORANCE FOR ALL OBJECTS
        plt.figure()
        plt.subplot(311)
        plt.title('Action ' + str(i) + ' ignorance')
        for j in range(len(data_array[0, 0, :])):
            line_color, marker, marker_points, label = get_details(data_array,
                                                                   i, j
                                                                   )
            plt.plot(np.arange(data_array.shape[0]), data_array[:, 2, j, i],
                     line_color
                     )
            plt.plot(marker_points, data_array[marker_points, 2, j, i],
                     marker, label=label
                     )

#        plt.legend()

        # PLOT PERCEPTRON OUTPUT FOR ALL OBJECTS
#        plt.figure()
        plt.subplot(312)
        plt.title('Action'+str(i)+' affordance prediction')
        for j in range(len(data_array[0, 0, :])):
            line_color, marker, marker_points, label = get_details(data_array,
                                                                   i, j
                                                                   )
            plt.plot(np.arange(data_array.shape[0]), data_array[:, 3, j, i],
                     line_color
                     )
            plt.plot(marker_points, data_array[marker_points, 3, j, i],
                     marker, label=label
                     )

#        plt.legend()

        # PLOT MOTIVATION SIGNAL FOR ALL OBJECTS
        plt.subplot(313)
        plt.title('Improvement prediction')
        for j in range(len(data_array[0, 0, :])):
            line_color, marker, marker_points, label = get_details(data_array,
                                                                   i, j
                                                                   )
            plt.plot(np.arange(data_array.shape[0]), data_array[:, 4, j, i],
                     line_color
                     )
            plt.plot(marker_points, data_array[marker_points, 4, j, i],
                     marker, label=label
                     )


if __name__ == '__main__':
    """Main"""
    # TESTS
#    plot('s1data_array.npy')
    get_average_data(10)
