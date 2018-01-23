#!/usr/bin/env python

"""
Main simulation script.

Here the main simulation of the system is run. Graphics are plotted for
the simulation if desired. This should be decided by button-clicks. One
button for starting/stopping the graphics. One button for showing
discrete steps in the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

from geometricshapes import Square, Circle
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


def goal_achievable_classifier(internal_fovea_image, external_fovea_image,
                               threshold):
    """
    Check if goal can be achieved by parameterised skill from current
    position.

    This is something that comes later with the parameterised skill
    implemented.

    FOR NOW: Just check if the right object is found in the external
    environment.

    Keyword arguments:
    - internal_fovea_image -- image array of internal fovea
    - external_fovea_image -- image array of external fovea
    - threshold -- float value threshold number (arbitrarily chosen)

    Compares the internal and external fovea images using the function
    check_images(). The foveas should be in different positions. If
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
    plt.pause(0.02)


def main():
    """
    Main simulation

    # FLAGS
    sub_goal = geometricshapes Object/None
    sub_goal_accomplished = True/False
    sub_goal_achievable = True/False
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
    FOR step = 1 to number_of_steps
        IF search_step >= max_search_steps
            SET search_step = 0 # Avoid endless search in external environment
        FUNCTION check_sub_goal() checks if sub_goal is found
        IF not sub_goal or search_step = 0
            FUNCTION foveate(internal_fovea) moves internal fovea
            FUNCTION check_sub_goal() checks if sub_goal is found
            # MAYBE THIS BELOW SHOULD BE OUTSIDE ANYWAY? JUST CHECK EXTERNAL
            # ENVIRONMENT IF A SUB-GOAL IS FOUND IN INTERNAL ENVIRONMENT?
            SET external_fovea position to match internal_fovea position
            IF sub_goal
                FUNCTION goal_accomplished_classifier() checks if
                    sub_goal_accomplished
        IF sub_goal and not sub_goal_accomplished
            search_step += 1
            FUNCTION foveate(external_rfovea) updates the position of
                external_fovea
            FUNCTION goal_achievable_classifier() checks if sub_goal_achievable
                from current position
            IF sub_goal_achievable
                FUNCTION parameterised_skill(x0, y0, x1, y1) moves polygon
                    in external environment using hand
                FUNCTION goal_accomplished_classifier checks if
                    sub_goal_accomplished
        IF sub_goal_accomplished
            SET sub_goal = None
            SET sub_goal_accomplished = False
            SET sub_goal_achievable = False

    # HERE GOES GRAPHICS/OUTPUT!
    """
    # SET VARIABLES
    number_of_steps = 100
    max_search_steps = 10
    search_step = 0
    accomplished_threshold = 0.01
    achievable_threshold = 0.01
    limits = np.array([[0.2, 0.8], [0.2, 0.8]])

    # FLAGS
    sub_goal = None
    sub_goal_accomplished = False
    sub_goal_achievable = False
    graphics_on = True

    # INITIALIZE ENVIRONMENT
    unit = 100  # SIZE OF SIDES OF ENVIRONMENT
    fovea_center = [0.5, 0.5]
    fovea_size = 0.2

    # INTERNAL ENVIRONMENT
    int_s1 = Square([0.35, 0.35], 0.15, [1, 0, 0], unit)
    int_c1 = Circle([0.65, 0.65], 0.15, [0, 1, 0], unit)
    int_objects = [int_s1, int_c1]

    int_env, int_fov, int_objects = environment.initialize(unit, fovea_center,
                                                           fovea_size,
                                                           int_objects
                                                           )

    # EXTERNAL ENVIRONMENT
    ext_s1 = Square([0.35, 0.65], 0.15, [1, 0, 0], unit)
    ext_c1 = Circle([0.65, 0.35], 0.15, [0, 1, 0], unit)
    ext_objects = [ext_s1, ext_c1]

    ext_env, ext_fov, ext_objects = environment.initialize(unit, fovea_center,
                                                           fovea_size,
                                                           ext_objects
                                                           )

    # PROVISORY GRAPHICS
    if graphics_on:
        plt.ion()
        plt.figure(1)
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
            sub_goal = False
        if not sub_goal:
            perception.foveate(int_fov, int_env)
#            perception.hard_foveate(int_fov, int_env, int_objects)
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
            perception.foveate(ext_fov, ext_env)
#            perception.hard_foveate(ext_fov, ext_env, ext_objects)
            ext_object = perception.check_sub_goal(ext_fov.center, ext_objects)
            sub_goal_achievable = goal_achievable_classifier(
                int_fov.get_focus_image(int_env),
                ext_fov.get_focus_image(ext_env),
                achievable_threshold
                )
            if sub_goal_achievable:
                actions.parameterised_skill(ext_object, int_fov.center, limits)
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
        if perception.check_images(int_env, ext_env, 0.00055):
            print('Goal accomplished!')
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
