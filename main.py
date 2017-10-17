#!/usr/bin/env python

"""Main simulation script"""

import numpy as np

def new_random_position(max_values):
    """
    Returns coordinates randomly chosen within limits
    
    Takes as input a list of dimensions of table object to make sure
    the new position is inside the working environment.
    """
    random_values = []
    for max_value in max_values:
        random_values.append(np.random.uniform(0, max_value))
    return np.array(random_values)

def move(object_, end_position):
    """
    Moves object from current position to end_position

    What about the hand? And fovea? Needs to be incorporated here. 
    Also, this should be implemented as a paremeterised skill later
    """
    object_.xy = end_position

def is_inside(fovea_coordinates, polygon):
    """
    Check if coordinates are inside object
    
    Uses shapely object function 'within'. Only checks for fovea in
    polygon for now, but could probably be generalised.
    """
    return poi(fovea_coordinates).within(poly(polygon.xy))

def sub_goal(fovea_coordinates, polygons):
    """
    Check if current position of fovea is on a sub-goal
    
    # PSEUDO-CODE (NOT FINISHED)
    FOR all polygons
        IF fovea is inside any polygon
            sub_goal == True
        ELSE
            sub_goal == False
    """
    sub_goal = False
    for polygon in polygons:
        if is_inside(fovea_coordinates, polygon):
            sub_goal = True    # Or just return True?
    return sub_goal    # Or just return False?

def goal_accomplished_classifier(internal_retina_matrix,
                                 external_retina_matrix
                                 ):
    """
    Compare the internal and external retina to see if goal is
    accomplished

    Takes two numpy arrays and compares the normalised distance between
    them.
    The retina arrays should be 2D.
    """
    internal_retina_vector = internal_retina_matrix.flatten() # Row-wise
    external_retina_vector = external_retina_matrix.flatten() # Row-wise
    diff = internal_retina_vector - external_retina_vector
    norm = np.linalg.norm(diff)
    return norm/len(internal_retina_vector)

def goal_achievable_classifier():
    """
    Checks if goal can be achieved by parameterised skill from
    current position. 
    This is something that comes later with the parameterised skill
    implemented.
    """
    return

def parameterised_skill():
    """
    Moves object in external environment to match internal goal. This
    parameterised skill has to learn. This will be the main part of the
    intelligent system. We will implement this using RBM.
    """

if __name__ == '__main__':
    """
    Main simulation

    # FLAGS
    sub_goal = True/False
    sub_goal_accomplished = True/False
    is_inside_ = True/False
    graphics_on = True/False    # Update graphics each step or not
    discrete_steps_on = True/False    # Whether the system should take discrete steps
    goal_achievable = True/False
    max_search_steps = integer

    # COUNTERS
    step
    search_step

    # PSEUDO-CODE OF MAIN FUNCTIONING
    SET goal_found = False
    SET goal_accomplished = False
    FOR step = 1 to number_of_steps
        IF search_step >= max_search_steps
            SET search_step = 0
        FUNCTION sub_goal() checks if sub_goal_found
        IF not sub_goal_found or search_step = 0
            FUNCTION foveate(internal_retina) moves internal retina
            FUNCTION sub_goal() checks if sub_goal_found
            # MAYBE THIS BELOW SHOULD BE OUTSIDE ANYWAY? JUST CHECK EXTERNAL
            # ENVIRONMENT IF A SUB-GOAL IS FOUND IN INTERNAL ENVIRONMENT? 
            SET external_retina position to match internal_retina position
            FUNCTION goal_accomplished_classifier() checks if goal_accomplished
        IF sub_goal_found and not goal_accomplished
            search_step += 1
            FUNCTION foveate(external_retina) updates the position of external
                retina
            FUNCTION goal_achievable_classifier(arguments) checks if
                goal_achievable from current position
            IF goal_achievable
                FUNCTION parameterised_skill(x0, y0, x1, y1) moves polygon
                    in external environment using hand
                FUNCTION goal_accomplished_classifier checks if
                    goal_accomplished
                IF goal_accomplished
                    goal_found = False
                    goal_accomplished = False
                    goal_achievable = False

    # HERE GOES GRAPHICS/OUTPUT! 
    """
    # Set variables
    number_of_steps = 10

    for step in range(1, number_of_steps+1):
        if not sub_goal_found:
            move(internal_retina, new_random_position(table_dimensions))
            sub_goal_found = sub_goal(internal_fovea, all_polygons)