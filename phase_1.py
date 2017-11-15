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
    OBJECT METHOD get_retina_image updates the fovea image  # NAME IS WRONG
    PERCEPTRON checks the knowledge about the focus image using
        (p.get_output)
    SET ignorance as the absolute distance from the knowledge to 1 or 0
    IF ignorance > average_ignorance
        FUNCTION parameterised_skill is applied on the object to try to
            move it to a random goal position  # THIS NEEDS CHECK_OVERLAP
        FUNCTION check_action_success checks if the action was
            successful by comparing the focus image before the action
            and the focus image at the target location after the action
            # THIS NEEDS TO ADD MASK TO IMAGES
        IF successful_action
            PERCEPTRON is updated (p.update_weights) with target = 1
        IF not successful_action
            PERCEPTRON is updated (p.update_weights) with target = 0
    FUNCTION update_avg_ignorance updates the average ignorance (leaky)
"""


def step(p):
    """Go through one step in the loop

    Keyword arguments:
    - p -- Perceptron object
    """
    pass


def update_avg_ignorance(average_ignorance, object_ignorance, rate=0.5):
    """Update the average ignorance

    Keyword arguments:
    - average_ignorance -- float of current average ignorance
    - object_ignorance -- float value of ignorance in current position
    - rate -- the update rate between 0 and 1 (float)

    Update the average ignorance using the formula:
    I_avg_t = I_avg_t-1 + rate * I_t,
    where I_avg_t-1 is the average ignorance before update, rate is the
    update rate and I_t is the current ignorance of the object in focus
    """
    change = -rate*average_ignorance + object_ignorance
    change = -(1-rate)*average_ignorance + rate*object_ignorance
    change = -rate*average_ignorance + (1-rate)*object_ignorance
    average_ignorance += change
    return average_ignorance


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
    pass


if __name__ == '__main__':
    """Main"""
