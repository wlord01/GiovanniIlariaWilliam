# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:45:14 2017

@author: William

Script for phase I of system.

In phase I the system should be able to form a sort of meta cognition.
This phase is driven by epistemic (associated to knowledge)/intrinsic
motivations. The system should log an average of the ignorance of the
objects in the environment. When observing an object, if the systems
ignorance of this object is higher than the average, then the system
wants to apply action to it to learn about it. If the ignorance is
lower than average it doesn't spend energy on it.

One way of defining successful application of skill is to "nail" some
objects to the environment so only some can be moved. Then the system
needs to learn what can be moved and what not. To make things more
interesting and complicated we can use scenarios where objects are
overlapping.

To evaluate if an action is successful or not the system should check
the start position and desired end position of the action before and
after the action is performed and then it can say if the action was
successful or not.


PSEUDO CODE

VARIABLES
- environment -- the environment image array
- fovea -- the fovea image array
- objects -- list of objects
- average_ignorance -- float number between 0 and 1 indicating the
    average ignorance of objects in the system
- ignorance -- float number between 0 and 1 indicating the ignorance
    of the object in focus
- number_of_steps -- int number of how many loops should be run
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
    IF ignorance + 0.05 > average_ignorance
        FUNCTION get_random_position() generates random xy coordinates
           as target position
        FUNCTION check_free_space() checks if the chosen target
            position is free
        FUNCTION parameterised_skill is applied on the object to try to
            move it to the target position
        FUNCTION check_action_success checks if the action was
            successful by comparing the focus image before the action
            and the focus image at the target location after the action
        IF successful_action
            PERCEPTRON is updated (p.update_weights) with target = 1
        IF not successful_action
            PERCEPTRON is updated (p.update_weights) with target = 0
    FUNCTION update_avg_ignorance updates the average ignorance (leaky)
"""


import numpy as np
import matplotlib.pyplot as plt

import simulation as s


def update_avg_ignorance(average_ignorance, object_ignorance, rate=0.05):
    """Update the average ignorance

    Keyword arguments:
    - average_ignorance -- float of current average ignorance
    - object_ignorance -- float value of ignorance in current position
    - rate -- the update rate between 0 and 1 (float)

    Update the average ignorance using the formula:
    I_avg_t = (1 - rate) * I_avg_t-1 + rate * I_t,
    where I_avg_t-1 is the average ignorance before update, rate is the
    leak rate and I_t is the current ignorance of the object in focus.
    """
    return (1-rate)*average_ignorance + rate*object_ignorance


def check_action_success(before_image, after_image):
    """Check the success of an action

    Keyword arguments:
    - before_image -- focus image array before action
    - after_image -- focus image array after action

    Compare the focus image array at the start point, before action
    attempt, and the focus image array at the end point of the action,
    after the action attempt with mask added to images.

    Return True/False.

    The mask is created by making all pixels black, except for the ones
    where the object in focus is in the before_image. This mask is then
    transferred to the after_image.
    """
    image_match = s.check_images(before_image, after_image)
    if image_match:
        return True
    else:
        return False


def check_target_position(environment, target_xy, fovea):
    """Return focus image at target positon

    Keyword arguments:
    - environment -- image array of environment
    - target_xy -- array of target position coordinates
    - fovea -- fovea object

    The function creates and returns a temporary focus image using the
    attributes of the real focus image and the target position.
    """
    temp_fovea = s.Fovea(target_xy, fovea.size, [1, 1, 1], fovea.unit)
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
    position and checks if it is all white.

    Returns True/False.
    """
    temp_image = check_target_position(environment, target_xy, fovea)
    if np.array_equal(temp_image, np.ones(temp_image.shape)):
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

    env = s.redraw_environment(env, unit, objects)
    fovea_im = fovea.get_focus_image(env)

    plt.subplot(121)
    plt.title('Training environment')
    plt.imshow(env)
    # PLOT DESK EDGES
    plt.plot([0.2*unit, 0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit],
             [0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit, 0.2*unit]
             )
    # PLOT FOVEA EDGES
    fov_indices = fovea.get_index_values()
    plt.plot([fov_indices[0][0], fov_indices[0][0], fov_indices[0][1],
              fov_indices[0][1], fov_indices[0][0]],
             [fov_indices[1][0], fov_indices[1][1], fov_indices[1][1],
              fov_indices[1][0], fov_indices[1][0]]
             )

    plt.subplot(122)
    plt.title('Focus image')
    plt.imshow(fovea_im)

    plt.draw()
    plt.pause(0.2)


def main():
    """Main simulation

    VARIABLES:
    - number_of_steps -- int number of how many loops should be run
    - p -- Perceptron object
    - environment -- image array of environment
    - fovea -- fovea object
    - objects -- list of objects in environment
    - average_ignorance -- float value of average ignorance
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
        IF ignorance + 0.05 > average_ignorance
            FUNCTION get_random_position() generates random xy coordinates
               as target position
            FUNCTION check_free_space() checks if the chosen target
                position is free
            FUNCTION parameterised_skill is applied on the object to try to
                move it to the target position
            FUNCTION check_action_success checks if the action was
                successful by comparing the focus image before the action
                and the focus image at the target location after the action
            IF successful_action
                PERCEPTRON is updated (p.update_weights) with target = 1
            IF not successful_action
                PERCEPTRON is updated (p.update_weights) with target = 0
        FUNCTION update_avg_ignorance updates the average ignorance (leaky)
    """
    # SET VARIABLES
    p = 1  # PERCEPTRON
    unit = 100
    average_ignorance = 0.5
    ignorance_bias = 0.05
    limits = np.array([[0.2, 0.8], [0.2, 0.8]])
    number_of_steps = 10

    env, fovea, objects = s.external_env_init(unit)

    graphics(env, fovea, objects, unit)

    for step in range(number_of_steps):
        s.hard_foveate(fovea, env, objects)
        graphics(env, fovea, objects, unit)
        fovea_im = fovea.get_focus_image(env)
        current_position = np.copy(fovea.center)
        current_knowledge = 0.5  # PUT PERCEPTRON HERE
        if current_knowledge < 0.5:
            current_ignorance = current_knowledge
        else:
            current_ignorance = 1 - current_knowledge
        if current_ignorance + ignorance_bias >= average_ignorance:
            before_image = np.copy(fovea_im)
            new_position = get_random_position(limits)
            fovea.move(new_position - fovea.center)
            graphics(env, fovea, objects, unit)
            while not check_free_space(env, new_position, fovea):
                new_position = get_random_position(limits)
                fovea.move(new_position - fovea.center)
                graphics(env, fovea, objects, unit)
            current_object = s.check_sub_goal(current_position, objects)
            s.parameterised_skill(current_object.center, new_position,
                                  current_object, limits)
            env = s.redraw_environment(env, unit, objects)
            target_pos_image = check_target_position(env, new_position,
                                                     fovea
                                                     )
            successful_action = check_action_success(before_image,
                                                     target_pos_image
                                                     )
            if successful_action:
                target = 1
                # UPDATE PERCEPTRON WEIGHTS HERE
            if not successful_action:
                target = 0
                # UPDATE PERCEPTRON WEIGHTS HERE
        average_ignorance = update_avg_ignorance(average_ignorance,
                                                 current_ignorance
                                                 )

        graphics(env, fovea, objects, unit)


if __name__ == '__main__':
    """Main"""
    main()

    # TESTS
#    image_1 = np.zeros((10, 10))
#    image_2 = np.ones((10, 10))
#
#    unit = 100
#    env, fovea, objects = s.external_env_init(unit)
#
#    fovea_im = fovea.get_focus_image(env)
#    print(check_action_success(fovea_im, fovea_im))
#
#    print(check_free_space(env, [0.65, 0.35], fovea.size, unit))

#    plt.imshow(fovea.get_focus_image(env))
#    plt.show()

#    average_ignorance = 0.5
#    object_ignorance = 0.5
#    print(average_ignorance)
#    average_ignorance = update_avg_ignorance(average_ignorance,
#                                             object_ignorance,
#                                             rate=0.05
#                                             )
#    print(average_ignorance)
#
#    print(get_random_position(np.array([[0.2, 0.8], [0.2, 0.8]])))
