#!/usr/bin/env python

"""
Phase 2 script.

Here the simulation of phase 2 is run. Graphics are plotted for the simulation
if desired (graphics_on = True/False). With this script we can run different
experiments to test the system on what it has learned in phase 1.
"""

import numpy as np
import matplotlib.pyplot as plt

from geometricshapes import Square, Circle
from perceptron import Perceptron
import actions
import environment
import perception


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


def goal_achievable_check(where_effect_predictors, what_effect_predictors,
                          goal_state, current_state, where_success_threshold,
                          what_success_threshold):
    """Check if goal is achievable by any action

    Keyword arguments:
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

    Uses forward model to check effects of actions on current state
    (external_focus_image). Compares (perception.check_images()) the
    predicted effects to the goal state (internal_focus_image) and
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
    for i in range(len(where_effect_predictors)):
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


def goal_achievable_classifier(successful_action):
    """
    Check if goal is achievable from current state

    Keyword arguments:
    - successful_action -- index of successful action (int)
    """
    if isinstance(successful_action, int):
        return True

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
    plt.imshow(int_env)
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
    plt.imshow(int_fov_im)

    ext_env = environment.redraw(ext_env, unit, ext_objects)
    ext_fov_im = ext_fov.get_focus_image(ext_env)

    plt.subplot(223)
    plt.title('External image')
    plt.xlim(0, unit)
    plt.ylim(0, unit)
    plt.imshow(ext_env)
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
    plt.imshow(ext_fov_im)

    plt.draw()
    plt.pause(0.2)


def main():
    """
    Main simulation

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
    """
    # SET VARIABLES
    number_of_steps = 100
    max_search_steps = 10
    search_step = 0
    accomplished_threshold = 0.01
    where_success_threshold = 0.01
    what_success_threshold = 0.005
    limits = np.array([[0.2, 0.8], [0.2, 0.8]])

    # FLAGS
    sub_goal = None
    sub_goal_accomplished = False
    sub_goal_achievable = False
    graphics_on = True

    # INITIALIZE ENVIRONMENT
    unit = 100  # SIZE OF SIDES OF ENVIRONMENT
    fovea_center = [0.54, 0.38]
    fovea_size = 0.2

    # INTERNAL ENVIRONMENT
    int_s1 = Square([0.3, 0.3], 0.15, [1, 0, 0], unit)
    int_c1 = Circle([0.7, 0.7], 0.15, [0, 0, 1], unit)
    int_s2 = Square([0.7, 0.3], 0.15, [0, 1, 0], unit)
    int_c2 = Circle([0.5, 0.5], 0.15, [1, 0, 0], unit)
    int_objects = [int_s1, int_c1, int_s2, int_c2]

    int_env, int_fov, int_objects = environment.initialize(unit, fovea_center,
                                                           fovea_size,
                                                           int_objects
                                                           )

    # EXTERNAL ENVIRONMENT
    ext_s1 = Square([0.3, 0.7], 0.15, [1, 0, 0], unit)
    ext_c1 = Circle([0.7, 0.7], 0.15, [0, 1, 0], unit)
    ext_s2 = Square([0.7, 0.3], 0.15, [0, 0, 1], unit)
    ext_c2 = Circle([0.4, 0.5], 0.15, [1, 0, 0], unit)
    ext_objects = [ext_s1, ext_c1, ext_s2, ext_c2]

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

        action_number = action_list.index(action)
        where_effect_predictor.read_weights_from_file('where_{}.npy'.format(
            str(action_number)))
        what_effect_predictor.read_weights_from_file('what_{}.npy'.format(
            str(action_number)))

        where_effect_predictors.append(where_effect_predictor)
        what_effect_predictors.append(what_effect_predictor)

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
#            perception.foveate(int_fov, int_env)
            perception.hard_foveate(int_fov, int_env, int_objects)
            ext_fov.move(int_fov.center - ext_fov.center)
            sub_goal = perception.check_sub_goal(int_fov.center,
                                                 int_objects
                                                 )
            if sub_goal:
                sub_goal_accomplished = goal_accomplished_classifier(
                    int_fov.get_focus_image(int_env),
                    ext_fov.get_focus_image(ext_env),
                    accomplished_threshold
                    )
        if sub_goal and not sub_goal_accomplished:
            search_step += 1
#            perception.foveate(ext_fov, ext_env)
            perception.hard_foveate(ext_fov, ext_env, ext_objects)
            ext_object = perception.check_sub_goal(ext_fov.center, ext_objects)
            ext_focus_image = ext_fov.get_focus_image(ext_env)
            int_focus_image = int_fov.get_focus_image(int_env)

            goal_state = np.array([np.concatenate(
                (int_fov.center, int_focus_image.flatten('F'))
                )]
                ).T
            current_state = np.array([np.concatenate(
                (ext_fov.center, ext_focus_image.flatten('F'))
                )]
                ).T

            successful_action = goal_achievable_check(
                where_effect_predictors,
                what_effect_predictors,
                goal_state,
                current_state,
                where_success_threshold,
                what_success_threshold
                )

            sub_goal_achievable = goal_achievable_classifier(successful_action)

            if sub_goal_achievable:
                action = action_list[successful_action]
                if action == actions.parameterised_skill:
                    action_input = (int_fov.center, limits)
                else:
                    action_input = ()

                action(ext_object, *action_input)
                ext_env = environment.redraw(ext_env, unit, ext_objects)
                ext_fov.move(int_fov.center - ext_fov.center)
                sub_goal_accomplished = goal_accomplished_classifier(
                    int_fov.get_focus_image(int_env),
                    ext_fov.get_focus_image(ext_env),
                    accomplished_threshold
                    )
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
        if perception.check_images(int_env, ext_env, 0.0000005):
            print('Goal accomplished at step {}!'.format(str(step)))
            break


if __name__ == '__main__':
    """
    Here we can run automated tests to check that everything works.

    After we made sure everything works we can just call main() here.

    Make sure to put plots in separate window (%matplotlib qt) to see
    graphics!
    """
    main()

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
#    plt.imshow(environment)
#
#    s2.move([0.2, 0.6])
#    plt.pause(2)
#
#    environment = redraw_environment(environment, unit, objects)
#
#    plt.imshow(environment)
