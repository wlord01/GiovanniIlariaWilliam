# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:02:22 2018

@author: William
"""

import numpy as np


def initialize_predictors(action_list, affordance_weights_file,
                          where_weights_file, what_weights_file, model_type,
                          unit=150, fovea_size=0.14):
    """Initialize predictors for testing"""
    from perceptron import Perceptron
    focus_image_side = int(unit * fovea_size)

    affordance_predictors = []

    affordance_predictor_input_shape = (focus_image_side**2 * 3, 1)
    affordance_predictor_output_shape = (1, 1)

    where_effect_predictors = []
    what_effect_predictors = []
    where_effect_predictor_output_shape = (2, 1)
    what_effect_predictor_input_shape = (focus_image_side**2 * 3, 1)
    what_effect_predictor_output_shape = (focus_image_side**2 * 3, 1)
    for action in action_list:
        if action == actions.parameterised_skill:
            where_effect_predictor_input_shape = np.array([4, 0])
        else:
            where_effect_predictor_input_shape = np.array([2, 0])

        where_effect_predictor = Perceptron(
            where_effect_predictor_input_shape,
            where_effect_predictor_output_shape,
            0,
            linear=True
            )
        what_effect_predictor = Perceptron(
            what_effect_predictor_input_shape,
            what_effect_predictor_output_shape,
            0,
#            binary=True
            )

        affordance_predictor = Perceptron(
            affordance_predictor_input_shape,
            affordance_predictor_output_shape,
            0
            )

        action_number = action_list.index(action)
        where_effect_predictor.read_weights_from_file(where_weights_file
            .format(action_number=str(action_number),
                    file_suffix=str(model_type)
                    )
            )
        what_effect_predictor.read_weights_from_file(what_weights_file
            .format(action_number=str(action_number),
                    file_suffix=str(model_type)
                    )
            )
        affordance_predictor.read_weights_from_file(affordance_weights_file
            .format(action_number=str(action_number),
                    file_suffix=str(model_type)
                    )
            )

        where_effect_predictors.append(where_effect_predictor)
        what_effect_predictors.append(what_effect_predictor)
        affordance_predictors.append(affordance_predictor)

    return (affordance_predictors, where_effect_predictors,
            what_effect_predictors)


def effect_predictors(where_effect_predictors, what_effect_predictors, unit,
                      fovea_size, object_size):
    # CHECK EFFECT PREDICTORS
    import numpy as np
    import matplotlib.pyplot as plt
    import environment

    object_images = environment.get_object_images(unit, fovea_size,
                                                  object_size
                                                  )
    what_effect_predictor = what_effect_predictors[1]
    where_effect_predictor = where_effect_predictors[1]
    for k in range(len(where_effect_predictors)):
        where_effect_predictor = where_effect_predictors[k]
        what_effect_predictor = what_effect_predictors[k]
        for i in object_images:
            image = i[2:]
            what_input = np.array([image]).T
#            end_position = np.array([0.2, 0.8])
            if k == 0:
                where_input = np.array([[0.2, 0.8, 0.5, 0.5]]).T
            else:
                where_input = np.array([[0.5, 0.5]]).T
            where_effect_predictor.set_input(where_input)
            what_effect_predictor.set_input(what_input)
            where_out = where_effect_predictor.get_output()
            what_out = what_effect_predictor.get_output()
            plt.figure()
            plt.subplot(121)
            plt.title(str(where_input))
            pixels = int(fovea_size * unit)
            plt.imshow(np.reshape(what_input, (pixels, pixels, 3), 'F'))
            plt.subplot(122)
            plt.title(str(where_out))
            plt.imshow(np.reshape(what_out, (pixels, pixels, 3), 'F'))


# FOR TESTING UTILITY PLANNER IN PHASE 2
# SET OUTPUT OF MAIN TO reward/utility AND INPUTS TO ACTION_ATTEMPTS
# AND THINK_TIME
#    ACTION_ATTEMPTS = 1
##    THINK_TIME = 5
#    print('Five objects with rewards 3, 3, 8 and 10')
#    for THINK_TIME in [0, 5, 10, 20, 30]:
#        rewards = []
#        for i in range(100):
#            reward = main(ACTION_ATTEMPTS, THINK_TIME)
#            if reward is not None:
#                rewards.append(reward)
#        average_reward = sum(rewards) / len(rewards)
#        s = 'Average reward {} for {} action, given {} steps to think'.format(
#                average_reward, ACTION_ATTEMPTS, THINK_TIME
#                )
#        print(s)

if __name__ == '__main__':
    # TESTS
    import actions
    model_type = 'IGN'
    where_weights_file = './Data/s0where_{action_number}_{file_suffix}.npy'
    what_weights_file = './Data/s0what_{action_number}_{file_suffix}.npy'
    affordance_weights_file = ('./Data/s0affordance_{action_number}_'
                               '{file_suffix}.npy'
                               )

    action_list = [actions.parameterised_skill, actions.activate,
                   actions.deactivate, actions.neutralize
                   ]
    (affordance_predictors,
     where_effect_predictors,
     what_effect_predictors
     ) = initialize_predictors(action_list, affordance_weights_file,
                               where_weights_file, what_weights_file,
                               model_type
                               )

#    effect_predictors(where_effect_predictors, what_effect_predictors, 150,
#                      0.14, 0.10
#                      )
