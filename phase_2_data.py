#!/usr/bin/env python

"""
Phase 2 data script

Contains functions for performance data acquisition and analysis of
phase 2 experiments.
"""

import numpy as np

import phase_2


def test_trial(model_type, trial_number, number_of_simulations):
    """Test performance of one trial

    Keyword arguments:
    - model_type (string) -- The model type (run in phase 1) to test
    - trial_number (string) -- The trial number (run in phase 1) to
      test
    - number_of_simulations (int) -- desired number of simulations

    Tests the goal based planner by running a desired number of
    simulations and returning data on goal accomplishment and
    number of required steps for goal based planning and data on reward
    for utility reasoning experiments. Set graphics_on to False in
    phase_2.py for much faster runtime.
    """
    trial_data = []
    for simulation_number in range(number_of_simulations):
        simulation_data = phase_2.main(model_type, trial_number)
        trial_data.append(simulation_data)

    return trial_data


if __name__ == '__main__':
    # TESTS
    data = test_trial('IGN', '6', 100)
