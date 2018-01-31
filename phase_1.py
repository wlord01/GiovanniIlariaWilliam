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
from geometricshapes import Square, Circle, Fovea, Rectangle
import actions
import environment
import perception


def leaky_average(average, current_value, leak_rate=0.05):
    """Update the overall ignorance

    Keyword arguments:
    - average -- float of current average value
    - current_value -- float of current value
    - leak_rate -- leak rate of update, between 0 and 1 (float)
    """
    return (1-leak_rate) * average + leak_rate * current_value


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


def get_ignorance(value):
    """Calculate ignorance

    Keyword arguments:
    - value -- current knowledge value (float)

    Return ignorance as the distance from the current knowledge to the
    closest of 0 or 1.
    """
    if value >= 0.5:
        ignorance = 1 - value
    else:
        ignorance = value

    return ignorance


def get_entropy(value):
    """Calculate entropy

    Keyword arguments:
    - value -- current knowledge value (float)

    Return ignorance as Shannon entropy of current knowledge.
    """
    return (- value * np.log2(value) - (1 - value) * np.log2(1 - value))


def select_action(action_list, improvement_predictors, focus_image):
    """Select action

    Keyword arguments:
    - action_list -- list of action functions
    - improvement_predictors -- list of corresponding improvement
      predictors
    - focus image -- the fovea image array

    Find which action has highest improvement prediction for object in
    focus. Return action number and corresponding improvement
    prediction.
    """
    improvement_predictions = []
    for action_number in range(len(action_list)):
        improvement_predictor = improvement_predictors[action_number]
        improvement_predictor.set_input(
            np.array([focus_image.flatten('F')]).T
            )
        improvement_prediction = improvement_predictor.get_output()
        improvement_predictions.append(abs(improvement_prediction))

    improvement_prediction = max(improvement_predictions)
    action_number = improvement_predictions.index(improvement_prediction)

    return action_number, improvement_prediction


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
    - unit -- Side size (int) of the square environment array
    - environment -- image array of environment
    - fovea -- fovea object
    - fovea_center -- position of fovea center (float)
    - fovea_size -- size of fovea (float)
    - objects -- list of objects in environment
    - late_objects -- list of late objects and the step number they
      are introduced (geomtericshape, int)
    - overall_ignorance -- float value of overall ignorance
    - leak_rate -- leak rate (float) of the overall ignorance
    - ignorance -- float number between 0 and 1 indicating the
      ignorance of the object in focus
    - ignorance_bias -- float value of bias added in ignorance
      comparison
    - limits -- array of coordinate limits [[x_min, x_max],
      [y_min, y_max]]
    - affordance_leak_rate -- learning rate (float) of affordance
      predictors
    - effect_learning_rate -- learning rate (float) of effect
      predictors

    FLAGS
    - action_performed -- True/False if action is performed
    - save_data -- Save data of simulation or not
    - plot_data -- Plot saved data or not
    - print_statements_on -- Print statements on or off at each step
    - graphics_on -- Simulation graphics or not

    FOR step in range number_of_steps
        FUNCTION hard_foveate(fovea, environment, objects) moves
        focus to an object
        OBJECT METHOD get_focus_image(environment) updates the fovea
            image
        FUNCTION select_action(action_list, improvement_predictors,
            focus_image) finds maximum improvement and returns action
            number and the corresponding improvement prediction
        FUNCTION affordance_predictor.set_input(input) sets afforance
            predictor input to the flattened focus image (image vector)
        FUNCTION affordance_predictor.get_output() gets affordance
            prediction output for the object in focus
        SET ignorance as Shannon entropy output of knowledge prediction
        IF improvement_prediction + selection_bias > overall_improvement
            IF action is move (parameterised_skill)
                WHILE generated target position is not free
                    FUNCTION get_random_position() generates random xy
                        coordinates as target position
                    FUNCTION check_free_space(environment, target_xy,
                        fovea) checks if the chosen target position is
                        free
            SET effect_predictor input (using target position if action
                is parameterised skill (move))
            EXECUTE action
            FUNCTION perception.check_effect(before_image, after_image)
                checks if the action had an effect by comparing the
                environment image array before and after action
            IF effect
                FUNCTION affordance_predictor.update_weights(1) updates
                    affordance predictor weights with target = 1
                FUNCTION hard_foveate(fovea, environment, objects)
                    moves focus to observed effect of action
                FUNCTION effect_predictor.update_weights(target)
                    updates the weights of the effect predictor using
                    position of fovea center and the flattened focus
                    image (image vector)
            IF not effect
                FUNCTION affordance_predictor.update_weights(0) updates
                    affordance predictor weights with target = 0
            FUNCTION improvement_predictor.update_weights(target)
                updates weights of improvement predictor using target
                -(H_after_action - H_before_action)
        FUNCTION leaky_average(overall_improvement,
            improvement_prediction, leak_rate) updates the overall
            improvement
        RESET action_performed to False
    """
    # SET VARIABLES
    unit = 100
    overall_improvement = 0
    selection_bias = 0.00001
    # TABLE X AND Y LIMITS IN ENVIRONMENT
    limits = np.array([[0.2, 0.8], [0.2, 0.8]])
    number_of_steps = 10000
    leak_rate = 0.3  # LEAKY INTEGRATOR
    affordance_learning_rate = 0.01
    improvement_learning_rate = 0.005
    where_effect_learning_rate = 0.1
    what_effect_learning_rate = 0.1
    improvement_predictor_weights = 0.00005
    rand_weights_init = 0.00075

    # FLAGS
    action_performed = False
    save_data = True
    plot_data = True
    print_statements_on = True
    graphics_on = False

    # INITIALIZE ENVIRONMENT
    fovea_center = [0.5, 0.5]
    fovea_size = 0.2

    s1 = Square([0.35, 0.65], 0.15, [1, 0, 0], unit)
    c1 = Circle([0.65, 0.35], 0.15, [0, 1, 0], unit)
    r1 = Rectangle([0., 0.], 0.15, [1, 0, 0], unit, 0)
#    s2 = Square([0.35, 0.35], 0.15, [0, 0, 1], unit, 0)
#    c2 = Circle([0., 0.], 0.15, [1, 0, 0], unit)
    objects = [s1, c1, r1]  # s2, c2]

    late_objects = np.array([[6000, r1]
                             ]
                            )

    env, fovea, objects = environment.initialize(unit, fovea_center,
                                                 fovea_size, objects
                                                 )

    fov_img_shape = np.array([fovea.get_focus_image(env).flatten('F')]).T.shape

    # ACTIONS
    action_list = [actions.parameterised_skill,
                   actions.activate,
                   actions.deactivate,
                   actions.neutralize
                   ]

    # PREDICTORS
    affordance_predictors = []
    where_effect_predictors = []
    what_effect_predictors = []
    improvement_predictors = []

    affordance_predictor_input_shape = fov_img_shape
    affordance_predictor_output_shape = (1, 1)
    where_effect_predictor_output_shape = (2, 1)
    what_effect_predictor_input_shape = fov_img_shape
    what_effect_predictor_output_shape = fov_img_shape
    improvement_predictor_input_shape = fov_img_shape
    improvement_predictor_output_shape = (1, 1)
    for action in action_list:
        if action == actions.parameterised_skill:
            where_effect_predictor_input_shape = np.array([4, 0])
        else:
            where_effect_predictor_input_shape = np.array([2, 0])

        affordance_predictors.append(Perceptron(
            affordance_predictor_input_shape,
            affordance_predictor_output_shape,
            affordance_learning_rate
            ))
        where_effect_predictors.append(Perceptron(
            where_effect_predictor_input_shape,
            where_effect_predictor_output_shape,
            where_effect_learning_rate,
            linear=True
            ))
        what_effect_predictors.append(Perceptron(
            what_effect_predictor_input_shape,
            what_effect_predictor_output_shape,
            what_effect_learning_rate,
#            binary=True
            ))
        improvement_predictor = Perceptron(improvement_predictor_input_shape,
                                           improvement_predictor_output_shape,
                                           improvement_learning_rate,
                                           linear=True
                                           )
        # WEIGHT INITIALISATION FOR IGNORANCE IM SIGNAL
        improvement_predictor.initialize_weights(improvement_predictor_weights)
#        improvement_predictor.initialize_rand_weights()
        # WEIGHT INITIALISATION FOR AFFORDANCE IM SIGNAL
        improvement_predictor.initialize_rand_sign_weights(rand_weights_init)

        improvement_predictors.append(improvement_predictor)

    if save_data:
        file_name = 'data_array.npy'
        object_images = environment.get_object_images(unit, fovea_size)
        number_of_objects = len(object_images)
        number_of_actions = len(action_list)
        types = [[0 for i in range(number_of_actions)]
                 for j in range(number_of_objects)]
        colors = [[0 for i in range(number_of_actions)]
                  for j in range(number_of_objects)]
        ignorance = [[0.5 for i in range(number_of_actions)]
                     for j in range(number_of_objects)]
        p_out = [[0.5 for i in range(number_of_actions)]
                 for j in range(number_of_objects)]
        motivation_signal = [[0 for i in range(number_of_actions)]
                             for j in range(number_of_objects)]
        features = [types, colors, ignorance, p_out, motivation_signal]
        number_of_features = len(features)
        data = np.zeros((number_of_steps,
                         number_of_features,
                         number_of_objects,
                         number_of_actions
                         )
                        )
        overall_improvement_data = []

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

        [action_number, improvement_prediction] = select_action(
            action_list,
            improvement_predictors,
            fovea_im
            )

        action = action_list[action_number]

        affordance_predictor = affordance_predictors[action_number]
        improvement_predictor = improvement_predictors[action_number]
        where_effect_predictor = where_effect_predictors[action_number]
        what_effect_predictor = what_effect_predictors[action_number]

        affordance_predictor.set_input(np.array([fovea_im.flatten('F')]).T)
        current_knowledge = affordance_predictor.get_output()

        current_ignorance = get_ignorance(current_knowledge)

        if improvement_prediction + selection_bias >= overall_improvement:
            action_performed = True

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

                where_effect_predictor_input = np.concatenate(
                    (np.array([new_position]).T,
                     np.array([old_position]).T
                     )
                    )
                what_effect_predictor_input = np.array(
                    [focus_image.flatten('F')]).T

                action_input = (new_position, limits)

            else:  # OTHER NON-PARAMETERISED ACTION
                where_effect_predictor_input = np.array([fovea.center]).T
                what_effect_predictor_input = np.array(
                    [focus_image.flatten('F')]).T

                action_input = ()

            p = np.random.rand()
            p = 1
            if p >= 0.3:
                action(current_object, *action_input)
            where_effect_predictor.set_input(where_effect_predictor_input)
            what_effect_predictor.set_input(what_effect_predictor_input)
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
                where_effect_predictor.update_weights(np.array(
                    [fovea.center]).T
                    )
                what_effect_predictor.update_weights(np.array(
                    [focus_image.flatten('F')]).T
                    )

            if not effect:
                target = 0
                affordance_predictor.update_weights(target)

            # UPDATE IMPROVEMENT PREDICTOR
            post_action_prediction = affordance_predictor.get_output()
            post_action_ignorance = get_ignorance(post_action_prediction)
            prediction_change = (post_action_prediction
                                 - current_knowledge)
            # COMMENT THE ROWs BELOW FOR USING CHANGE IN AFFORDANCE PRED
#            prediction_change = - (post_action_ignorance
#                                   - current_ignorance)

            improvement_predictor.update_weights(prediction_change)

            old_overall_improvement = overall_improvement
            overall_improvement = leaky_average(
                overall_improvement,
                abs(prediction_change),
                leak_rate
                )

        if print_statements_on:
            print('Step ', step)
            print(('Object {}').format(str(objects.index(current_object))))
            print(('Action {}').format(str(action_number)))
            print(('1st predictor output: {}').format(str(current_knowledge)))
            print(('Improvement prediction: {} vs overall: {}').format(
                  str(improvement_prediction), str(old_overall_improvement))
                  )
            print(('Actual improvement: {}').format(str(prediction_change)))
            if action_performed:
                print('Action performed')
            else:
                print('Action not performed')

        if not action_performed:
            overall_improvement = leaky_average(overall_improvement,
                                                0,
                                                leak_rate
                                                )

        action_performed = False

        if graphics_on:
            graphics(env, fovea, objects, unit)

        if save_data:
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
                obj_ign = get_ignorance(out)
                ignorance[object_number][action_number] = obj_ign
                p_out[object_number][action_number] = out
                improvement_predictor.set_input(
                    np.array([image.flatten('F')]).T
                    )
                im_pred = improvement_predictor.get_output()
                motivation_signal[object_number][action_number] = abs(im_pred)

            for i in range(len(features)):
                data[step, i] = features[i]

        if save_data:
            overall_improvement_data.append(overall_improvement[0])

    if save_data:
        np.save(file_name, data)

    if save_data and plot_data:
        phase_1_data.plot(file_name)
        plt.figure()
        plt.plot(overall_improvement_data)

    for p in where_effect_predictors:
        file_name = 'where_{}.npy'.format(
            str(where_effect_predictors.index(p))
            )
        p.write_weights_to_file(file_name)
    for p in what_effect_predictors:
        file_name = 'what_{}.npy'.format(
            str(what_effect_predictors.index(p))
            )
        p.write_weights_to_file(file_name)
    for p in affordance_predictors:
        file_name = 'affordance_{}.npy'.format(
            str(affordance_predictors.index(p))
            )
        p.write_weights_to_file(file_name)

    # CHECK EFFECT PREDICTORS
#    import tests
#    tests.effect_predictors(where_effect_predictors, what_effect_predictors,
#                            unit, fovea_size)

if __name__ == '__main__':
    """Main"""
    main()
    print('END')
