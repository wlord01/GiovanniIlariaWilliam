# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:45:14 2017

@author: William

Script for phase I of system.

In phase I the system should be able to form a sort of meta cognition.
This phase is driven by epistemic (associated to knowledge)/intrinsic
motivations. The system should log an overall ignorance of the objects
in the environment. When observing an object, if the systems ignorance
of this object is higher than the average, then the system wants to
apply action to it to learn about it. If the ignorance is lower than
average it doesn't spend energy on it.

One way of defining successful application of skill is to "nail" some
objects to the environment so only some can be moved. Then the system
needs to learn what can be moved and what not. To make things more
interesting and complicated we can use scenarios where objects are
overlapping.

To evaluate if an action is successful or not the system should check
effects of the performed action.


PSEUDO CODE

VARIABLES
- environment -- the environment image array
- fovea -- the fovea image array
- objects -- list of objects
- overall_ignorance -- float number between 0 and 1 indicating the
    overall ignorance of objects in the system
- ignorance -- float number between 0 and 1 indicating the ignorance
    of the object in focus
- number_of_steps -- int number of how many loops should be run
- target -- int value (0, 1) for supervised update of perceptron

FLAGS
- effect -- True/False effect after applied action

FOR step in range number_of_steps
    FUNCTION hard_foveate(fovea, environment, objects) moves
    focus to an object
    OBJECT METHOD get_focus_image updates the fovea image
    PERCEPTRON checks the knowledge about the focus image using
        (p.get_output)
    SET ignorance as the absolute distance from the knowledge to 1 or 0
    IF ignorance + 0.05 > overall_ignorance
        FUNCTION get_random_position() generates random xy coordinates
           as target position
        FUNCTION check_free_space() checks if the chosen target
            position is free
        FUNCTION parameterised_skill is applied on the object to try to
            move it to the target position
        FUNCTION check_effect() checks if the action had an effect by
            comparing the environment image array after and before
            action
        IF effect
            PERCEPTRON is updated (p.update_weights) with target = 1
        IF not effect
            PERCEPTRON is updated (p.update_weights) with target = 0
    FUNCTION update_overall_ignorance updates the overall ignorance
        (leaky integrator)
"""


import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
import phase_1_data
from geometricshapes import Square, Circle, Fovea
import actions
import environment
import perception


def update_overall_ignorance(overall_ignorance, object_ignorance, rate=0.05):
    """Update the overall ignorance

    Keyword arguments:
    - overall_ignorance -- float of current overall ignorance
    - object_ignorance -- float value of ignorance in current position
    - rate -- the update rate between 0 and 1 (float)

    Update the overall ignorance using the formula:
    I_ovrl_t = (1 - rate) * I_ovrl_t-1 + rate * I_t,
    where I_ovrl_t-1 is the overall ignorance before update, rate is the
    leak rate and I_t is the current ignorance of the object in focus.
    """
    return (1-rate)*overall_ignorance + rate*object_ignorance


def check_target_position(environment, target_xy, fovea):
    """Return focus image at target positon

    Keyword arguments:
    - environment -- image array of environment
    - target_xy -- array of target position coordinates
    - fovea -- fovea object

    The function creates and returns a temporary focus image using the
    attributes of the real focus image and the target position.
    """
    temp_fovea = Fovea(target_xy, fovea.size, [0, 0, 0], fovea.unit)
    temp_image = temp_fovea.get_focus_image(environment)
    return temp_image


def check_free_space(environment, target_xy, fovea):
    """Check if target area is free

    Keyword arguments:
    - env_image -- image array of the environment
    - target_xy -- the xy coordinates of the target position
    - fovea -- fovea object

    Check if the focus area around the target position enx_xy is free
    space. The method creates a temporary fovea image at the target
    position and checks if it contains only zeros.

    Returns True/False.
    """
    temp_image = check_target_position(environment, target_xy, fovea)
    if np.array_equal(temp_image, np.zeros(temp_image.shape)):
        return True
    else:
        return False


def get_random_position(limits):
    """Generate random xy coordinates within limits

    Keyword arguments:
    - limits -- array of [[x_min, x_max], [y_min, y_max]]
    """
    x = (limits[0][1]-limits[0][0])*np.random.random_sample() + limits[0][0]
    y = (limits[1][1]-limits[1][0])*np.random.random_sample() + limits[1][0]
    return np.array([x, y])


def graphics(env, fovea, objects, unit):
    """Provisory function for plotting the graphics of the system.

    Keyword arguments:
    - env -- image array of the environment
    - fovea -- fovea object
    - objects -- a list containing the objects in the environment
    - unit -- the size of the sides of the quadratic environment
    """
    plt.clf()

    env = environment.redraw(env, unit, objects)
    fovea_im = fovea.get_focus_image(env)

    plt.subplot(121)
    plt.title('Training environment')
    plt.xlim(0, unit)
    plt.ylim(0, unit)
    plt.imshow(env)

    # PLOT DESK EDGES
    plt.plot([0.2*unit, 0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit],
             [0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit, 0.2*unit], 'w-'
             )

    # PLOT FOVEA EDGES
    fov_indices = fovea.get_index_values()
    plt.plot([fov_indices[0][0], fov_indices[0][0], fov_indices[0][1],
              fov_indices[0][1], fov_indices[0][0]],
             [fov_indices[1][0], fov_indices[1][1], fov_indices[1][1],
              fov_indices[1][0], fov_indices[1][0]], 'w-'
             )

    plt.subplot(122)
    plt.title('Focus image')
    plt.imshow(fovea_im)

    plt.draw()
    plt.pause(0.01)


def main():
    """Main simulation

    VARIABLES:
    - number_of_steps -- int number of how many loops should be run
    - p -- Perceptron object
    - environment -- image array of environment
    - fovea -- fovea object
    - objects -- list of objects in environment
    - overall_ignorance -- float value of overall ignorance
    - ignorance -- float number between 0 and 1 indicating the ignorance
      of the object in focus
    - ignorance_bias -- float value of bias added in ignorance
      comparison
    - limits -- array of coordinate limits [[x_min, x_max],
      [y_min, y_max]]
    - target -- int value (0, 1) for supervised update of perceptron

    FLAGS
    - successful_action -- True/False success of action

    FOR step in range number_of_steps
        FUNCTION hard_foveate(fovea, environment, objects) moves
        focus to an object
        OBJECT METHOD get_focus_image updates the fovea image
        PERCEPTRON checks the knowledge about the focus image using
            (p.get_output)
        SET ignorance as the absolute distance from the knowledge to 1 or 0
        IF ignorance + 0.05 > overall_ignorance
            FUNCTION get_random_position() generates random xy coordinates
               as target position
            FUNCTION check_free_space() checks if the chosen target
                position is free
            FUNCTION parameterised_skill is applied on the object to try to
                move it to the target position
            FUNCTION check_effect() checks if the action had an effect by
                comparing the environment image array after and before
                action
            IF effect
                PERCEPTRON is updated (p.update_weights) with target = 1
            IF not effect
                PERCEPTRON is updated (p.update_weights) with target = 0
        FUNCTION update_overall_ignorance updates the overall ignorance
            (leaky integrator)
    """
    # SET VARIABLES
    unit = 100
    overall_ignorance = 1
    ignorance_bias = 0.
    # TABLE X AND Y LIMITS IN ENVIRONMENT
    limits = np.array([[0.2, 0.8], [0.2, 0.8]])
    number_of_steps = 1000
    leak_rate = 0.2  # LEAKY INTEGRATOR
    affordance_learning_rate = 0.025
    effect_learning_rate = 0.01

    # FLAGS
    move_made = False

    save_data = True
    plot_data = True
    print_statements_on = True
    graphics_on = False

    # INITIALIZE ENVIRONMENT
    fovea_center = [0.5, 0.5]
    fovea_size = 0.2

    s1 = Square([0.35, 0.65], 0.14, [1, 0, 0], unit)
    c1 = Circle([0.65, 0.35], 0.14, [0, 1, 0], unit)
#    s2 = Square([0.35, 0.35], 0.14, [0, 0, 1], unit, 0)
    c2 = Circle([0., 0.], 0.14, [1, 0, 0], unit)
    objects = [s1, c1, c2]  # s2, c2]

    late_objects = np.array([[200, c2]
                             ]
                            )

    env, fovea, objects = environment.initialize(unit, fovea_center,
                                                 fovea_size, objects
                                                 )

    fov_img_shape = np.array([fovea.get_focus_image(env).flatten('F')]).T.shape

    # ACTIONS
    action_list = [actions.parameterised_skill,
                   actions.activate,
                   actions.deactivate
                   ]

    # PREDICTORS
    affordance_predictor_1 = Perceptron(fov_img_shape, (1, 1),
                                        affordance_learning_rate)
    affordance_predictor_2 = Perceptron(fov_img_shape, (1, 1),
                                        affordance_learning_rate)
    affordance_predictor_3 = Perceptron(fov_img_shape, (1, 1),
                                        affordance_learning_rate)

    affordance_predictors = [affordance_predictor_1,
                             affordance_predictor_2,
                             affordance_predictor_3
                             ]

    effect_predictor_1 = Perceptron(np.array([4, 0]) + fov_img_shape,
                                    np.array([2, 0]) + fov_img_shape,
                                    effect_learning_rate
                                    )
    effect_predictor_2 = Perceptron(np.array([2, 0]) + fov_img_shape,
                                    np.array([2, 0]) + fov_img_shape,
                                    effect_learning_rate
                                    )
    effect_predictor_3 = Perceptron(np.array([2, 0]) + fov_img_shape,
                                    np.array([2, 0]) + fov_img_shape,
                                    effect_learning_rate
                                    )

    effect_predictors = [effect_predictor_1,
                         effect_predictor_2,
                         effect_predictor_3
                         ]

    if save_data:
        file_name = 'data_array.npy'
        object_images = environment.get_object_images(unit, fovea_size)
        number_of_objects = len(object_images)
        number_of_actions = len(action_list)
        types = [[0 for i in range(number_of_actions)]
                 for j in range(number_of_objects)]
        colors = [[0 for i in range(number_of_actions)]
                  for j in range(number_of_objects)]
        ignorance = [[1 for i in range(number_of_actions)]
                     for j in range(number_of_objects)]
        p_out = [[0.5 for i in range(number_of_actions)]
                 for j in range(number_of_objects)]
        features = [types, colors, ignorance, p_out]
        number_of_features = len(features)
        data = np.zeros((number_of_steps,
                         number_of_features,
                         number_of_objects,
                         number_of_actions
                         )
                        )

    if graphics_on:
        graphics(env, fovea, objects, unit)

    # MAIN LOOP
    for step in range(number_of_steps):
        if np.any(late_objects == step):
            for i in np.where(late_objects == step)[0]:
                print('Introduce new object!')
                late_object = late_objects[i, 1]
                position = get_random_position(limits)
                while not check_free_space(env, position, fovea):
                    position = get_random_position(limits)
                late_object.center += position
            env = environment.redraw(env, unit, objects)

        perception.hard_foveate(fovea, env, objects)

        if graphics_on:
            graphics(env, fovea, objects, unit)

        fovea_im = fovea.get_focus_image(env)
        current_position = np.copy(fovea.center)
        current_object = perception.check_sub_goal(current_position, objects)

        action = np.random.choice(action_list)  # RANDOM FOR NOW
#        action = action_list[0]
        affordance_predictor = affordance_predictors[action_list.index(action)]
        effect_predictor = effect_predictors[action_list.index(action)]

        affordance_predictor.set_input(np.array([fovea_im.flatten('F')]).T)
        current_knowledge = affordance_predictor.get_output()

        # SHANNON ENTROPY
        current_ignorance = (- current_knowledge * np.log2(current_knowledge) -
                             (1-current_knowledge) *
                             np.log2(1-current_knowledge))

        if current_ignorance + ignorance_bias >= overall_ignorance:
            move_made = True

            # PERFORM ACTION AND CHECK EFFECT
            env_before_action = np.copy(env)
            focus_image = fovea.get_focus_image(env)

            if action == actions.parameterised_skill:
                old_position = np.copy(fovea.center)
                new_position = get_random_position(limits)
                fovea.move(new_position - fovea.center)

                if graphics_on:
                    graphics(env, fovea, objects, unit)

                while not check_free_space(env, new_position, fovea):
                    new_position = get_random_position(limits)
                    fovea.move(new_position - fovea.center)

                    if graphics_on:
                        graphics(env, fovea, objects, unit)

                effect_predictor_input = np.concatenate(
                    (np.array([new_position]).T,
                     np.array([old_position]).T,
                     np.array([focus_image.flatten('F')]).T
                     )
                    )

                actions.parameterised_skill(current_object.center,
                                            new_position,
                                            current_object,
                                            limits
                                            )
            else:  # OTHER NON-PARAMETERISED ACTION
                effect_predictor_input = np.concatenate(
                    (np.array([fovea.center]).T,
                     np.array([focus_image.flatten('F')]).T
                     )
                    )

                action(current_object)

            effect_predictor.set_input(effect_predictor_input)
            env = environment.redraw(env, unit, objects)

            effect = perception.check_effect(env_before_action, env)

            if effect:
                target = 1
                affordance_predictor.update_weights(target)

                perception.hard_foveate(fovea,
                                        (env - env_before_action).clip(0, 1),
                                        objects
                                        )
                focus_image = fovea.get_focus_image(env)
                effect_predictor.update_weights(
                    np.concatenate((np.array([fovea.center]).T,
                                    np.array([focus_image.flatten('F')]).T
                                    )
                                   )
                    )

            if not effect:
                target = 0
                affordance_predictor.update_weights(target)

        if graphics_on:
            graphics(env, fovea, objects, unit)

        if save_data:
            action_number = action_list.index(action)
            for object_number in range(len(object_images)):
                object_type = int(object_images[object_number][0])
                object_color = int(object_images[object_number][1])
                types[object_number] = [object_type for i in
                                        types[object_number]
                                        ]
                colors[object_number] = [object_color for i in
                                         colors[object_number]
                                         ]

                image = object_images[object_number][2:]
                affordance_predictor.set_input(
                    np.array([image.flatten('F')]).T
                    )
                out = affordance_predictor.get_output()
                obj_ign = (- out * np.log2(out) - (1-out) * np.log2(1-out))
                ignorance[object_number][action_number] = obj_ign
                p_out[object_number][action_number] = out

            data[step, 0] = types
            data[step, 1] = colors
            data[step, 2] = ignorance
            data[step, 3] = p_out

        if print_statements_on:
            print('Step ', step)
            print(('Ignorance  {} vs overall {}').format(
                  str(current_ignorance), str(overall_ignorance))
                  )
            if move_made:
                print('Move attempt on object #{}'.format(
                      str(objects.index(current_object) + 1))
                      )
            else:
                print('No move attempt on object #{}'.format(
                      str(objects.index(current_object) + 1))
                      )

        move_made = False

        overall_ignorance = update_overall_ignorance(overall_ignorance,
                                                     current_ignorance,
                                                     leak_rate
                                                     )

    if save_data:
        np.save(file_name, data)

    if plot_data:
        phase_1_data.plot(file_name)

if __name__ == '__main__':
    """Main"""
    main()
    print('END')
