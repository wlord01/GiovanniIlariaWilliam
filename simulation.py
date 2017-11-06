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

from geometricshapes import Square, Circle, Retina


def new_random_position(max_values):
    """
    Return coordinates randomly chosen within limits.

    Take as input a list of dimensions of table object to make sure
    the new position is inside the working environment.
    """
    random_values = []
    for max_value in max_values:
        random_values.append(np.random.uniform(0, max_value))
    return np.array(random_values)


def is_inside(coordinates, object_):
    """
    Check if coordinates are inside object.

    Takes input xy-coordinates and polygon object.
    Return True/False.
    """
    if object_.is_inside(coordinates):
        return True
    else:
        return False


def sub_goal(fovea_coordinates, polygons):
    """
    Check if current position of fovea is on a sub-goal.

    If the fovea is inside a polygon a sub-goal is found.

    # PSEUDO-CODE (NOT FINISHED)
    found = False
    FOR all polygons
        IF not found
            FUNCTION is_inside(fovea_coordinates, polygon) checks if
                fovea is inside the polygon
            IF fovea is inside polygon
                SET found = True
    IF found
        return True
    ELSE
        return False
    """
    found = False
    for polygon in polygons:
        if not found:
            if is_inside(fovea_coordinates, polygon):
                found = True
                return polygon

    if found:
        return True
    else:
        return False


def goal_accomplished_classifier(internal_retina_matrix,
                                 external_retina_matrix, threshold
                                 ):
    """
    Compare the internal and external retina to see if goal is
    accomplished.

    Takes two numpy arrays and compares the normalised distance between
    their flattened versions. The retina arrays should be 2D. The
    threshold is arbitrarily chosen and set, to determine if the
    flattened vectors are similar enough (external image close enough
    to internal goal image).
    """
    internal_retina_vector = internal_retina_matrix.flatten()  # Row-wise
    external_retina_vector = external_retina_matrix.flatten()  # Row-wise
    diff = internal_retina_vector - external_retina_vector
    norm = np.linalg.norm(diff)
    if norm/len(internal_retina_vector) <= threshold:
        return True
    else:
        return False


def goal_achievable_classifier(int_sub_goal, ext_object):
    """
    Check if goal can be achieved by parameterised skill from current
    position.

    This is something that comes later with the parameterised skill
    implemented.

    FOR NOW: Just check if the right object is found in the external
    environment.

    Keyword arguments:
    - int_sub_goal -- found internal sub goal object
    - ext_object -- found external object

    IF sub_goal_found in both internal and external environment
        IF object.type_, object.size and object.color is same for both
            RETURN True
        ELSE
            RETURN False
    ELSE
        RETURN False
    """
    if int_sub_goal and ext_object:
        if (int_sub_goal.type_ == ext_sub_goal.type_ and
                int_sub_goal.color == ext_sub_goal.color and
                int_sub_goal.size == ext_sub_goal.size):
            return True
        else:
            return False
    else:
        return False


def parameterised_skill(ext_fov, int_fov, object_):
    """
    Move object in external environment.

    This parameterised skill has to learn. This will be the main part of
    the intelligent system. We will implement this using RBM.

    FOR NOW: Implement hardwired moving of objects.

    Arguments:
    - ext_fov -- array of float coordinates of external fovea
    - int_fov -- array of float coordinates of internal fovea
    - object_ -- the object that the function should move
    """
    vector = int_fov - ext_fov
    move_object(object_, vector)


def move_object(object_, vector):
    """Move the objects center along the vector.

    Needs to know limits so object is not moved outside table. This
    check is inside the class method move() now, but maybe it should
    be out here?
    """
    object_.move(vector)


def foveate(retina):
    """
    Foveate retina.

    Uses the {RGB --> Black/White --> Add noise --> (smooth) -->
    foveate} procedure.

    For now: just move retina to random pos.
    """
    retina.move(0.8*np.random.random_sample(2) - 0.4)


def main():
    """
    Main simulation

    # FLAGS
    sub_goal_found = True/False
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
    SET sub_goal_found = False
    SET sub_goal_accomplished = False
    SET sub_goal_achievable = False
    FOR step = 1 to number_of_steps
        IF search_step >= max_search_steps
            SET search_step = 0 # Avoid endless search in external environment
        FUNCTION sub_goal() checks if sub_goal_found
        IF not sub_goal_found or search_step = 0
            FUNCTION foveate(internal_retina) moves internal retina
            FUNCTION sub_goal() checks if sub_goal_found
            # MAYBE THIS BELOW SHOULD BE OUTSIDE ANYWAY? JUST CHECK EXTERNAL
            # ENVIRONMENT IF A SUB-GOAL IS FOUND IN INTERNAL ENVIRONMENT?
            SET external_retina position to match internal_retina position
            FUNCTION goal_accomplished_classifier() checks if
                sub_goal_accomplished
        IF sub_goal_found and not sub_goal_accomplished
            search_step += 1
            FUNCTION foveate(external_retina) updates the position of
                external_retina
            FUNCTION goal_achievable_classifier() checks if sub_goal_achievable
                from current position
            IF sub_goal_achievable
                FUNCTION parameterised_skill(x0, y0, x1, y1) moves polygon
                    in external environment using hand
                FUNCTION goal_accomplished_classifier checks if
                    sub_goal_accomplished
                IF sub_goal_accomplished
                    SET sub_goal_found = False
                    SET sub_goal_accomplished = False
                    SET sub_goal_achievable = False

    # HERE GOES GRAPHICS/OUTPUT!
    """
    # SET VARIABLES
    number_of_steps = 10
    sub_goal_found = False
    sub_goal_accomplished = False
    sub_goal_achievable = False

    # MAIN FUNCTIONING
    for step in range(1, number_of_steps+1):
        if not sub_goal_found:
            move(internal_retina, new_random_position(table_dimensions))
            sub_goal_found = sub_goal(internal_fovea, all_polygons)


if __name__ == '__main__':
    """
    Here we can run automated tests to check that everything works.

    After we made sure everything works we can just call main() here.
    """
    # main()
    # Run tests

    # Set up environment
    unit = 100

    # INTERNAL
    int_image = np.ones([unit, unit, 3])

    # Create objects
    int_c = Circle([0.3, 0.4], 0.15, [1, 0, 0], unit)
    int_s = Square([0.6, 0.6], 0.1, [0, 0, 1], unit)
    int_objects = [int_c, int_s]

    int_c.draw(int_image)
    int_s.draw(int_image)

    # Get retina
    int_retina = Retina([0.55, 0.55], 0.3, [1, 1, 1], unit)
    int_ret_image = int_retina.get_retina_image(int_image)

    plt.imshow(int_image)
    plt.plot([20, 20, 80, 80, 20], [20, 80, 80, 20, 20])
    ret_indices = int_retina.get_index_values()
    print(ret_indices)
    plt.plot([ret_indices[0][0], ret_indices[0][0], ret_indices[0][1],
              ret_indices[0][1], ret_indices[0][0]],
             [ret_indices[1][0], ret_indices[1][1], ret_indices[1][1],
              ret_indices[1][0], ret_indices[1][0]]
             )

    plt.figure()
    plt.imshow(int_ret_image)
    plt.show()

    int_sub_goal = sub_goal(int_retina.center, int_objects)
    print(int_sub_goal)

    # EXTERNAL
    ext_s = Square([0.4, 0.4], 0.1, [0, 0, 1], unit)
    ext_objects = [ext_s]
    ext_ret = Retina([0.45, 0.4], 0.3, [1, 1, 1], unit)
    ext_sub_goal = sub_goal(ext_ret.center, ext_objects)
    print(ext_sub_goal)

    print(goal_achievable_classifier(int_sub_goal, ext_sub_goal))
