# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:15:38 2017

@author: William

Functions:
- get_average_weights(number_of_simulations, model_type) - Averages
  the predictor weights over the simulations specified by
  number_of_simulations and model_type. Saves averaged weighs to file.
- get_end_predictions(number_of_simulations, model_type) - Gets the
  final predicted action success probability from all simulations
  defined by number_of_simulations and model_type and saves to file.
- plot_bar_charts(number_of_simulations) - Gets files produced by
  get_end_predictions, calculates means and standard deviations for
  each object and plots these in bar charts for each model and each
  action.
- get_average_data(number_of_simulations, model_type) - Averages the
  data over the number of simulations defined by the function argument
  number_of_simulations. Argument model_type choses which files to use.
- plot(file_name) - plots the ignorance and predictor outputs for the
  objects.
"""


import matplotlib.pyplot as plt
import numpy as np


def get_average_weights(number_of_simulations, model_type):
    """Get average predictor weights

    Keyword arguments:
    - number_of_simulations -- Number of simulations (int) over which
      to calculate average data.
    - model_type -- Model type name/data file suffix (string). This
      can be 'IGN', 'FIX' or 'IMP' for ignorance motivation signal,
      fix threshold version or improvement signal version.

    Gets weights from all simulations, calculates mean of these and
    saves to new file. This is done for all predictors (affordance,
    where and what) for each action, for the model type.
    """
    number_of_actions = 4
    predictor_types = ['affordance', 'where', 'what']
    for predictor_type in predictor_types:
        for action_number in range(number_of_actions):
            file_name = 'Data/s1{}_{}_{}.npy'.format(
                str(predictor_type), str(action_number), str(model_type)
                )
            simulation_weights = np.load(file_name)
            predictor_weights = np.empty((number_of_simulations,
                                          simulation_weights.shape[0],
                                          simulation_weights.shape[1]
                                          ), dtype=float
                                         )
            predictor_weights[0] = simulation_weights
            for simulation_number in range(2, number_of_simulations + 1):
                file_name = 'Data/s{}{}_{}_{}.npy'.format(
                    str(simulation_number), str(predictor_type),
                    str(action_number), str(model_type)
                    )
                simulation_weights = np.load(file_name)
                predictor_weights[simulation_number - 1] = simulation_weights

            average_weights = np.mean(predictor_weights, 0)
            file_name = 'Data/{}sim_average_{}_{}_{}.npy'.format(
                str(number_of_simulations), str(predictor_type),
                str(action_number), str(model_type)
                )

            np.save(file_name, average_weights)


def get_end_predictions(number_of_simulations, model_type):
    """Get average end affordance prediction

    Keyword arguments:
    - number_of_simulations -- Number of simulations (int) over which
      to calculate average data.
    - model_type -- Model type name/data file suffix (string). This
      can be 'IGN', 'FIX' or 'IMP' for ignorance motivation signal,
      fix threshold version or improvement signal version.

    Concatenates the final affordance predictions from all the
    simulations and saves to files. These can be used for plotting
    bar charts on mean final predictions.
    """
    number_of_objects = 9
    number_of_actions = 4
    final_predictions = np.empty((number_of_simulations, number_of_objects,
                                  number_of_actions
                                  ), dtype=float
                                 )

    for i in range(1, number_of_simulations + 1):
        file_name = './Data/s{}data_array_{}.npy'.format(str(i),
                                                         str(model_type)
                                                         )
        simulation_data = np.load(file_name)
        final_predictions[i - 1] = simulation_data[-1, 3]

    file_name = './Data/{}sim_average_end_prediction_{}'.format(
        str(number_of_simulations), str(model_type)
        )

    np.save(file_name, final_predictions)


def plot_bar_charts(number_of_simulations=10):
    """Plot bar charts for end predictions

    Keyword arguments:
    - number_of_simulations - number of simulations (int) to average
      over.

    Calculates the mean final action success predictions and standard
    deviations for each object, for each model and for each action.
    Plots bar charts for each action with bars for each model, showing
    the mean final affordance prediction for each object.

    FOR each action
      GET final predictions
      CALC mean and stdev
      PLOT bars
    """
    number_of_objects = 9
    number_of_actions = 4

    names = ['O{}'.format(i) for i in range(1, number_of_objects + 1)]
    w = 0.3

    for action in range(number_of_actions):
        plt.figure()

        model_type = 'IGN'
        file_name = 'Data/{}sim_average_end_prediction_{}.npy'.format(
            str(number_of_simulations), str(model_type)
            )
        final_predictions = np.load(file_name)
        mean_predictions = np.mean(final_predictions, 0)
        stdev = np.std(final_predictions, 0)
        plt.bar(np.arange(0, number_of_objects) - w,
                mean_predictions[:, action], width=w, color='r',
                yerr=stdev[:, action], ecolor='black', capsize=5,
                align='center', tick_label=names
                )

        model_type = 'FIX'
        file_name = 'Data/{}sim_average_end_prediction_{}.npy'.format(
            str(number_of_simulations), str(model_type)
            )
        final_predictions = np.load(file_name)
        mean_predictions = np.mean(final_predictions, 0)
        stdev = np.std(final_predictions, 0)
        plt.bar(np.arange(0, number_of_objects), mean_predictions[:, action],
                width=w, color='g', yerr=stdev[:, action], ecolor='black',
                capsize=5, align='center', tick_label=names
                )

        model_type = 'IMP'
        file_name = 'Data/{}sim_average_end_prediction_{}.npy'.format(
            str(number_of_simulations), str(model_type)
            )
        final_predictions = np.load(file_name)
        mean_predictions = np.mean(final_predictions, 0)
        stdev = np.std(final_predictions, 0)
        plt.bar(np.arange(0, number_of_objects) + w,
                mean_predictions[:, action], width=w, yerr=stdev[:, action],
                ecolor='black', capsize=5, align='center'
                )


def get_average_data(number_of_simulations, model_type='IMP'):
    """Calculate average data over many simulations

    Keyword_arguments:
    - number_of_simulations -- Number of simulations (int) over which
      to calculate average data
    - model_type -- Model type name/data file suffix (string). This
      can be 'IGN', 'FIX' or 'IMP' for ignorance motivation signal,
      fix threshold version or improvement signal version.

    Concatenates the data from all the simulations and calculates
    average values for ignorance and predictor outputs at each time
    step over the simulations. A data file is created and saved with
    the average data.
    """
    all_data = np.load('./Data/s1data_array_{}.npy'.format(str(model_type)))

    for i in range(2, number_of_simulations + 1):
        file_name = './Data/s{}data_array_{}.npy'.format(str(i),
                                                         str(model_type)
                                                         )
        simulation_data = np.load(file_name)
        all_data[:, 2:5] += simulation_data[:, 2:5]

    average_data = all_data
    average_data[:, 2:5] = average_data[:, 2:5] / number_of_simulations
    average_data_file_name = './Data/{}sim_average_data_{}.npy'.format(
        str(number_of_simulations),
        str(model_type)
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
#    plot('Data/s0data_array_IGN.npy')
#    get_average_data(10)
#    get_end_predictions(10, model_type='IGN')
#    get_end_predictions(10, model_type='FIX')
#    get_end_predictions(10, model_type='IMP')
#    plot_bar_charts(10)
    get_average_weights(10, 'IGN')
