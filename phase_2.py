#!/usr/bin/env python

"""
Phase 2 script.

Here the simulation of phase 2 is run. Graphics are plotted for the simulation
if desired (graphics_on = True/False). With this script we can run different
experiments to test the system on what it has learned in phase 1.
"""

import numpy as np
import matplotlib.pyplot as plt

from geometricshapes import Square, Circle, Rectangle
from perceptron import Perceptron
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


def goal_accomplished_classifier(internal_fovea_image, external_fovea_image,
                                 threshold
                                 ):
    """
    Check if sub goal is accomplished.

    Keyword arguments:
    - internal_fovea_image -- image array of internal fovea
    - external_fovea_image -- image array of external fovea
    - threshold -- float value threshold number (arbitrarily chosen)

    Compares the internal and external fovea images using the function
    check_images(). The foveas should have the same position. If
    images are equal enough (within threshold), the goal is
    accomplished.
    """
    same_images = perception.check_images(internal_fovea_image,
                                          external_fovea_image,
                                          threshold
                                          )
    if same_images:
        return True
    else:
        return False


def afforded_actions_check(current_state, affordance_predictors, threshold):
    """
    Check which actions are afforded

    Keyword arguments:
    - current_state -- current state of the system (fovea coordinates
      and focus image)
    - affordance_predictors -- affordance predictors (Perceptron
      objects)
    - threshold -- float number above which the affordance prediction
      has to be for the action to be considered afforded

    Checks with the affordance predictors which actions are afforded in
    the current state by comparing affordance prediction to threshold.
    """
    afforded_actions = []
    for predictor in affordance_predictors:
        predictor.set_input(current_state[2:])
        prediction = predictor.get_output()
        if prediction >= threshold:
            afforded_actions.append(affordance_predictors.index(predictor))

    return afforded_actions


def goal_achievable_check(afforded_actions, where_effect_predictors,
                          what_effect_predictors, goal_state, current_state,
                          where_success_threshold, what_success_threshold):
    """Check if goal is achievable by any action

    Keyword arguments:
    - afforded_actions -- integer index numbers of afforded actions in
      current state
    - where_effect_predictors -- Where effect predictors (list of
      Perceptron objects)
    - what_effect_predictors -- What effect predictors (list of
      Perceptron objects)
    - goal_state -- Center of internal fovea (float array with shape
      (2, 1)) and internal focus image array (int array) concatenated
    - current_state -- Center of external fovea (float array with shape
      (2, 1)) and external focus image array (int array) concatenated
    - where_success_threshold -- threshold (float) for successful
      comparison of position
    - what_succes_threshold -- threshold (float) for successful
      comparison of images

    Output:
    - int or None

    Uses forward model to check effects of afforded actions on current
    state (external_focus_image). Compares (perception.check_images())
    the predicted effects to the goal state (internal_focus_image) and
    returns action index number if there is a match. Returns None if no
    predicted effect matches the goal state.

    FOR all actions
        IF action is move
            where_predictor_input <-- [goal_position,
                                       current_position]
        ELSE
            where_predictor_input <-- current_position
        what_predictor_input <-- current state focus image

        where_prediction <-- where_predictor.get_output()
        what_prediction <-- what_predictor.get_output()

        where_success <-- distance(where_prediction,
                                   goal_position
                                   ) <= where_threshold
        what_success <-- perception.check_images(goal_focus_image,
                                                 current_focus_image,
                                                 image_threshold
                                                 )
        IF where_success and what_success
            return index_of_action

        return None
    """
    for i in afforded_actions:
        where_predictor = where_effect_predictors[i]
        what_predictor = what_effect_predictors[i]

        if i == 0:
            where_input = np.concatenate((goal_state[0:2], current_state[0:2]))
        else:
            where_input = current_state[0:2]

        where_predictor.set_input(where_input)

        what_input = current_state[2:]
        what_predictor.set_input(what_input)

        where_prediction = where_predictor.get_output()
        what_prediction = what_predictor.get_output()

        where_difference = np.linalg.norm((where_prediction - goal_state[0:2]))
        where_success = where_difference <= where_success_threshold

        what_success = perception.check_images(goal_state[2:],
                                               what_prediction,
                                               what_success_threshold
                                               )

        if where_success and what_success:
            return i

    return None


def goal_achievable_classifier(successful_action, free_space):
    """
    Check if goal is achievable from current state

    Keyword arguments:
    - successful_action -- index of successful action (int)

    Goal is achievable if there is a successful action and, in case the
    successful action is the move action, if the target position is
    free.
    """
    if isinstance(successful_action, int):
        if successful_action == 0 and free_space:
            return True
        elif successful_action == 0 and not free_space:
            return False
        else:
            return True
    else:
        return False


def graphics(int_env, int_objects, int_fov, ext_env, ext_objects, ext_fov,
             unit):
    """Provisory function for plotting the graphics of the system.

    Keyword arguments:
    - int_env -- the image array of the internal environment
    - int_objects -- a list containing the objects in the internal
      environment
    - int_fov -- the fovea object in the internal environment
    - ext_env -- the image array of the external environment
    - ext_objects -- a list containing the objects in the external
      environment
    - ext_fov -- the fovea object in the external environment
    - unit -- the size of the sides of the quadratic environment
    """
    plt.clf()

    int_env = environment.redraw(int_env, unit, int_objects)
    int_fov_im = int_fov.get_focus_image(int_env)

    plt.subplot(221)
    plt.title('Internal image')
    plt.xlim(0, unit)
    plt.ylim(0, unit)
    plt.axis('off')
    plt.imshow(int_env, origin='lower')
    # PLOT DESK EDGES
    plt.plot([0.2*unit, 0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit],
             [0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit, 0.2*unit], 'w-'
             )
    # PLOT FOVEA EDGES
    fov_indices = int_fov.get_index_values()
    plt.plot([fov_indices[0][0], fov_indices[0][0], fov_indices[0][1],
              fov_indices[0][1], fov_indices[0][0]],
             [fov_indices[1][0], fov_indices[1][1], fov_indices[1][1],
              fov_indices[1][0], fov_indices[1][0]], 'w-'
             )

    plt.subplot(222)
    plt.title('Internal fovea')
    plt.axis('off')
    plt.imshow(int_fov_im, origin='lower')

    ext_env = environment.redraw(ext_env, unit, ext_objects)
    ext_fov_im = ext_fov.get_focus_image(ext_env)

    plt.subplot(223)
    plt.title('External image')
    plt.xlim(0, unit)
    plt.ylim(0, unit)
    plt.axis('off')
    plt.imshow(ext_env, origin='lower')
    # PLOT DESK EDGES
    plt.plot([0.2*unit, 0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit],
             [0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit, 0.2*unit], 'w-'
             )
    # PLOT FOVEA EDGES
    fov_indices = ext_fov.get_index_values()
    plt.plot([fov_indices[0][0], fov_indices[0][0], fov_indices[0][1],
              fov_indices[0][1], fov_indices[0][0]],
             [fov_indices[1][0], fov_indices[1][1], fov_indices[1][1],
              fov_indices[1][0], fov_indices[1][0]], 'w-'
             )

    plt.subplot(224)
    plt.title('External fovea')
    plt.axis('off')
    plt.imshow(ext_fov_im, origin='lower')

    plt.draw()
    plt.pause(0.2)


def main(model_type, trial_number):
    """
    Main simulation

    Keyword arguments:
    - model_type -- String object, IGN/FIX/IMP as file suffix in weight
      file names.
    - trial_number -- String object used for defining which weight
      files to load from saved phase 1 simulations.

    Returns tuple of binary goal accomplished variable, completion step
    and mean number of explored actions per step. If utility reasoning
    is on, the acquired utility/reward is returned instead.

    # FLAGS
    sub_goal = geometricshapes Object/None
    sub_goal_accomplished = True/False
    sub_goal_achievable = True/False
    successful_action = Action index number (int)/None
    max_search_steps = integer
    number_of_steps = integer
    graphics_on = True/False    # Update graphics each step or not
    discrete_steps_on = True/False

    # COUNTERS
    step
    search_step

    # PSEUDO-CODE OF MAIN FUNCTIONING
    SET sub_goal = None
    SET sub_goal_accomplished = False
    SET sub_goal_achievable = False

    sub_goal <-- check_sub_goal(internal_fovea_center, internal_objects)
    IF sub_goal
        sub_goal_accomplished <-- goal_accomplished_classifier(
            internal_focus_image,
            external_focus_image,
            accomplished_threshold
            )

    FOR step = 1 to number_of_steps
        IF search_step >= max_search_steps
            search_step <-- 0 # Avoid endless search in external environment
            sub_goal <-- None
        IF not sub_goal
            FUNCTION foveate(internal_fovea) moves internal fovea
            external_fovea.center <-- internal_fovea.center
            sub_goal <-- check_sub_goal(internal_fovea_center,
                                        internal_objects
                                        )
            IF sub_goal
                sub_goal_accomplished <-- goal_accomplished_classifier(
                    internal_focus_image,
                    external_focus_image,
                    accomplished_threshold
                    )
        IF sub_goal and not sub_goal_accomplished
            search_step <-- search_step + 1
            FUNCTION foveate(external_fovea) updates the position of
                external_fovea
            successful_action <-- goal_achievable_check(
                where_effect_predictors,
                what_effect_predictors,
                goal_state,
                current_state,
                where_success_threshold,
                what_success_threshold
                )
            sub_goal_achievable <-- goal_achievable_classifier(
                successful_action
                )
            IF sub_goal_achievable
                action <-- action_list[successful_action]
                action(object, *action_input) performs successful action
                sub_goal_accomplished <-- goal_accomplished_classifier(
                    internal_focus_image,
                    external_focus_image,
                    accomplished_threshold
                    )
        IF sub_goal_accomplished
            SET sub_goal = None
            SET sub_goal_accomplished = False
            SET sub_goal_achievable = False

        # HERE GOES GRAPHICS/OUTPUT!
        IF graphics_on
            FUNCTION graphics(internal_environment, internal_objects,
                              internal_focus_image, external_environment,
                              external_objects, external_focus_image, unit
                              ) plots the graphics of the simulation

    If utility_reasoning_on is set to True the initial part of the
    simulation will run the following algorithm for the number of steps
    defined by THINKING_STEPS:

    sub_goal <-- SCAN(goal_image)
    current_state <-- SCAN(environment)
    goal_achievable <-- GOAL_ACHIEVABLE_CHECK(current_state)
    IF goal_achievable
        current_utility <-- COMPUTE_UTILITY(affordance_prediction,
                                            goal_utility
                                            )
        sub_goal <-- False
        goal_accomplished <-- False
        goal_achievable <-- False
    IF current_utility >= overall_utility
        overall_utility <-- LEAKY_AVERAGE(overall_utility, current_utility)

    After this thinking phase an acting phase is run for the number of
    actions defined by ACTION_ATTEMPTS, according to the following
    algorithm:

    IF not sub_goal
        sub_goal <-- SCAN(goal_image)
        goal_accomplished <-- GOAL_ACCOMPLISHED_CHECK(goal_image, environment)
    IF sub_goal and not goal_accomplished
        current_state <-- SCAN(environment)
        goal_achievable <-- GOAL_ACHIEVABLE_CHECK(current_state)
        IF goal_achievable
            current_utility <-- COMPUTE_UTILITY(affordance_prediction, goal_)
            IF current_utility >= overall_utility
                ACTION(current_state)
                goal_accomplished <-- GOAL_ACCOMPLISHED_CHECK(goal_image,
                                                              environment
                                                              )
            IF goal_accomplished
                sub_goal <-- False
                goal_accopmlished <-- False
                goal_achievable <-- False
    overall_utility <-- LEAKY_AVERAGE(overall_utility, 0)

    """
    # SET CONSTANTS
    unit = 150  # SIZE OF SIDES OF ENVIRONMENT
    fovea_size = 0.14
    object_size = 0.10
    number_of_steps = 125
    max_search_steps = 8
    THINKING_STEPS = 10
    ACTION_ATTEMPTS = 3
    accomplished_threshold = 0.01
    where_success_threshold = 0.01
    what_success_threshold = 0.0035
    limits = np.array([[0.2, 0.8], [0.2, 0.8]])
    where_weights_file = ('./Data/s{trial_number}where_{action_number}_'
                          '{file_suffix}.npy'
                          )
    what_weights_file = ('./Data/s{trial_number}what_{action_number}_'
                         '{file_suffix}.npy'
                         )
    affordance_weights_file = ('./Data/s{trial_number}affordance_'
                               '{action_number}_{file_suffix}.npy'
                               )

    # SET VARIABLES
    fovea_center = [0.54, 0.38]
    search_step = 0
    overall_utility = 0
    actions_made = 0
    reward = 0
    explored_actions = 0

    # FLAGS
    sub_goal = None
    sub_goal_accomplished = False
    sub_goal_achievable = False
    graphics_on = False
    utility_reasoning_on = True
    restricted_search_on = True  # Toggle restricted forward model search

    # INITIALIZE INTERNAL ENVIRONMENT
    int_s1 = Square([0.2, 0.5], object_size, [0, 1, 0], unit, 1)
    int_r1 = Rectangle([0.8, 0.2], object_size, [1, 0, 0], unit, 0, 2)
    int_s2 = Square([0.2, 0.2], object_size, [1, 0, 0], unit, 4)
    int_c1 = Circle([0.5, 0.8], object_size, [0, 0, 1], unit, 1)
    int_c2 = Circle([0.5, 0.2], object_size, [1, 0, 0], unit, 10)
    int_objects = [int_s1, int_r1, int_s2, int_c1]  # , int_c2]

    int_env, int_fov, int_objects = environment.initialize(unit, fovea_center,
                                                           fovea_size,
                                                           int_objects
                                                           )

    # INITIALIZE EXTERNAL ENVIRONMENT
    ext_s1 = Square([0.2, 0.5], object_size, [0, 0, 1], unit)
    ext_r1 = Rectangle([0.8, 0.2], object_size, [0, 1, 0], unit, 0)
    ext_s2 = Square([0.2, 0.8], object_size, [1, 0, 0], unit)
    ext_c1 = Circle([0.5, 0.8], object_size, [0, 1, 0], unit)
    ext_c2 = Circle([0.5, 0.5], object_size, [1, 0, 0], unit)
    ext_objects = [ext_s1, ext_r1, ext_s2, ext_c1]  # , ext_c2]

    ext_env, ext_fov, ext_objects = environment.initialize(unit, fovea_center,
                                                           fovea_size,
                                                           ext_objects
                                                           )

    # ACTIONS
    action_list = [actions.parameterised_skill,
                   actions.activate,
                   actions.deactivate,
                   actions.neutralize
                   ]

    # PREDICTORS
    focus_image = int_fov.get_focus_image(int_env)
    focus_image_shape = np.array([focus_image.flatten('F')]).T.shape

    affordance_predictors = []

    affordance_predictor_input_shape = focus_image_shape
    affordance_predictor_output_shape = (1, 1)

    where_effect_predictors = []
    what_effect_predictors = []
    where_effect_predictor_output_shape = (2, 1)
    what_effect_predictor_input_shape = focus_image_shape
    what_effect_predictor_output_shape = focus_image_shape
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
            .format(trial_number=str(trial_number),
                    action_number=str(action_number),
                    file_suffix=str(model_type)
                    )
            )
        what_effect_predictor.read_weights_from_file(what_weights_file
            .format(trial_number=str(trial_number),
                    action_number=str(action_number),
                    file_suffix=str(model_type)
                    )
            )
        affordance_predictor.read_weights_from_file(affordance_weights_file
            .format(trial_number=str(trial_number),
                    action_number=str(action_number),
                    file_suffix=str(model_type)
                    )
            )

        where_effect_predictors.append(where_effect_predictor)
        what_effect_predictors.append(what_effect_predictor)
        affordance_predictors.append(affordance_predictor)

    # PROVISORY GRAPHICS
    if graphics_on:
        plt.ion()
        plt.figure()
        plt.axis('off')

        graphics(int_env, int_objects, int_fov, ext_env, ext_objects, ext_fov,
                 unit
                 )

    # MAIN FUNCTIONING
    sub_goal = perception.check_sub_goal(int_fov.center, int_objects)
    if sub_goal:
        sub_goal_accomplished = goal_accomplished_classifier(
            int_fov.get_focus_image(int_env),
            ext_fov.get_focus_image(ext_env),
            accomplished_threshold
            )

    for step in range(1, number_of_steps+1):
        if search_step >= max_search_steps:
            search_step = 0
            sub_goal = None
        if not sub_goal:
#            perception.foveate(int_fov, int_env, int_objects)
            perception.hard_foveate(int_fov, int_env, int_objects)
            ext_fov.move(int_fov.center - ext_fov.center)
            sub_goal = perception.check_sub_goal(int_fov.center,
                                                 int_objects
                                                 )

            if graphics_on:
                graphics(int_env, int_objects, int_fov, ext_env, ext_objects,
                         ext_fov, unit
                         )

            if sub_goal:
                sub_goal_accomplished = goal_accomplished_classifier(
                    int_fov.get_focus_image(int_env),
                    ext_fov.get_focus_image(ext_env),
                    accomplished_threshold
                    )
        if sub_goal and not sub_goal_accomplished:
            search_step += 1
#            perception.foveate(ext_fov, ext_env, ext_objects)
            perception.hard_foveate(ext_fov, ext_env, ext_objects)
            ext_object = perception.check_sub_goal(ext_fov.center, ext_objects)
            ext_focus_image = ext_fov.get_focus_image(ext_env)
            int_focus_image = int_fov.get_focus_image(int_env)

            if graphics_on:
                graphics(int_env, int_objects, int_fov, ext_env, ext_objects,
                         ext_fov, unit
                         )

            goal_state = np.array([np.concatenate(
                (int_fov.center, int_focus_image.flatten('F'))
                )]
                ).T
            current_state = np.array([np.concatenate(
                (ext_fov.center, ext_focus_image.flatten('F'))
                )]
                ).T

            if restricted_search_on:
                afforded_actions = afforded_actions_check(
                    current_state, affordance_predictors, threshold=0.5
                    )
            else:
                afforded_actions = [i for i in range(len(action_list))]

            explored_actions += len(afforded_actions)

            successful_action = goal_achievable_check(
                afforded_actions,
                where_effect_predictors,
                what_effect_predictors,
                goal_state,
                current_state,
                where_success_threshold,
                what_success_threshold
                )

            free_space = perception.check_free_space(ext_env, goal_state[0:2],
                                                     ext_fov
                                                     )
            sub_goal_achievable = goal_achievable_classifier(successful_action,
                                                             free_space
                                                             )

            if sub_goal_achievable:
                if utility_reasoning_on:
                    affordance_predictor = affordance_predictors[
                        successful_action
                        ]
                    affordance_predictor_input = np.array(
                        [ext_focus_image.flatten('F')]
                        ).T
                    affordance_predictor.set_input(affordance_predictor_input)
                    success_prediction = affordance_predictor.get_output()
                    current_utility = success_prediction * sub_goal.value

                if utility_reasoning_on and step <= THINKING_STEPS:
                    if current_utility >= overall_utility:
                        overall_utility = leaky_average(overall_utility,
                                                        current_utility,
                                                        leak_rate=1.0
                                                        )
                    sub_goal = None
                    sub_goal_accomplished = False
                    sub_goal_achievable = False
                    search_step = 0
                else:
                    if (not utility_reasoning_on or
                            current_utility >= overall_utility):
                        action = action_list[successful_action]
                        if action == actions.parameterised_skill:
                            action_input = (int_fov.center, limits)
                        else:
                            action_input = ()

                        action(ext_object, *action_input)
                        ext_env = environment.redraw(ext_env, unit,
                                                     ext_objects
                                                     )
                        ext_fov.move(int_fov.center - ext_fov.center)
                        sub_goal_accomplished = goal_accomplished_classifier(
                            int_fov.get_focus_image(int_env),
                            ext_fov.get_focus_image(ext_env),
                            accomplished_threshold
                            )

                        actions_made += 1
                        if sub_goal_accomplished:
                            reward += sub_goal.value
                    else:
                        sub_goal = None
                        sub_goal_accomplished = False
                        sub_goal_achievable = False

                    overall_utility = leaky_average(overall_utility, 0,
                                                    leak_rate=0.1)

        if sub_goal_accomplished:
            sub_goal = None
            sub_goal_accomplished = False
            sub_goal_achievable = False
            search_step = 0

        if graphics_on:
            graphics(int_env, int_objects, int_fov, ext_env, ext_objects,
                     ext_fov, unit
                     )

        # BREAK IF GOAL IMAGE IS ACCOMPLISHED
        if perception.check_images(int_env, ext_env, 1e-4):
#            print('Goal accomplished at step {}!'.format(str(step)))
            return (1, step, explored_actions / step, reward)
        elif actions_made >= ACTION_ATTEMPTS:
            return (0, step, explored_actions / step, reward)

    return (0, step, explored_actions / step, reward)


if __name__ == '__main__':
    """
    Here we can run automated tests to check that everything works.

    After we made sure everything works we can just call main() here.

    Make sure to put plots in separate window (%matplotlib qt) to see
    graphics!
    """
    model_type = 'IMPs'
    trial_number = '1'
    runs = 20
    trials = ['1', '2', '3', '4', '5', '8', '9']
#    np.random.seed(12)

    # TEST UTILITY ACCUMULATION
    utility = np.zeros((len(trials), runs))
    for trial_number in range(len(trials)):
        for i in range(runs):
            data = main(model_type, trials[trial_number])
            utility[trial_number, i] = data[3]
    trial_means = np.mean(utility, axis=1)
    system_mean = np.mean(trial_means)
    system_SEM = np.std(trial_means) / np.sqrt(len(trials))

    # TEST GOAL ACCOMPLISHING
#    total_complete_runs = 0
#    for trial_number in ['1','2','3','4','5','6','7','8','9','10']:        
#        complete_runs = 0
#        for i in range(runs):
#            data = main(model_type, '1')
#            if data[0] == 1:
#                complete_runs += 1
#        print('Completion ratio: {}/{}'.format(str(complete_runs), str(runs)))
#        total_complete_runs += complete_runs
#    print('Total completion ratio: {}/{}'.format(str(total_complete_runs), str(1000)))
#
#    print('Average complete runs: ', str(total_complete_runs/10))
    # Run tests

#    plt.clf()
#
#    # Set up environment
#    unit = 100
#    fovea_center = [0.5, 0.5]
#    fovea_size = 0.2
#
#    # Create objects
#    c1 = Circle([0.3, 0.4], 0.15, [1, 0, 0], unit)
#    s1 = Square([0.6, 0.6], 0.15, [0, 0, 1], unit)
#    s2 = Square([0., 0.], 0.15, [0, 1, 0], unit)
#    objects = [c1, s1, s2]
#
#    environment, fovea, objects = initialize_environment(unit, fovea_center,
#                                                         fovea_size, objects
#                                                         )
#
#    plt.figure(1)
#    plt.imshow(environment, origin='lower')
#
#    s2.move([0.2, 0.6])
#    plt.pause(2)
#
#    environment = redraw_environment(environment, unit, objects)
#
#    plt.imshow(environment, origin='lower')
