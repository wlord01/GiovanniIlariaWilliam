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


def check_sub_goal(fovea_coordinates, polygons):
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


def check_images(image_array_1, image_array_2, threshold):
    """Compare two image arrays and say if they show the same thing.

    Takes two numpy arrays and calculates the normalised distance
    between their flattened versions. The threshold is arbitrarily
    chosen and set, to determine if the flattened vectors are similar
    enough.
    """
    vector_1 = image_array_1.flatten()  # Row-wise
    vector_2 = image_array_2.flatten()  # Row-wise
    diff = vector_1 - vector_2
    norm = np.linalg.norm(diff)
    if norm/len(vector_1) <= threshold:
        return True
    else:
        return False


def goal_accomplished_classifier(internal_retina_image, external_retina_image,
                                 threshold
                                 ):
    """
    Check if sub goal is accomplished.

    Keyword arguments:
    - internal_retina_image -- image array of internal retina
    - external_retina_image -- image array of external retina
    - threshold -- float value threshold number (arbitrarily chosen)

    Compares the internal and external retina images using the function
    check_images(). The retinas should have the same position. If
    images are equal enough (within threshold), the goal is
    accomplished.
    """
    same_images = check_images(internal_retina_image,
                               external_retina_image,
                               threshold
                               )
    if same_images:
        return True
    else:
        return False


def goal_achievable_classifier(internal_retina_image, external_retina_image,
                               threshold):
    """
    Check if goal can be achieved by parameterised skill from current
    position.

    This is something that comes later with the parameterised skill
    implemented.

    FOR NOW: Just check if the right object is found in the external
    environment.

    Keyword arguments:
    - internal_retina_image -- image array of internal retina
    - external_retina_image -- image array of external retina
    - threshold -- float value threshold number (arbitrarily chosen)

    Compares the internal and external retina images using the function
    check_images(). The retinas should be in different positions. If
    images are equal enough (within threshold), the goal is
    accomplished.
    """
    same_images = check_images(internal_retina_image,
                               external_retina_image,
                               threshold
                               )

    if same_images:
        return True
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
    retina.move(0.3*np.random.random_sample(2) - 0.15)


def internal_env_init(unit):
    """Initiate the internal environment.

    Keyword arguments:
    - unit -- the size of the sides of the quadratic environment
    """
    int_env = np.ones([unit, unit, 3])
    int_ret = Retina([0.5, 0.5], 0.2, [1, 1, 1], unit)
    int_s1 = Square([0.35, 0.35], 0.15, [1, 0, 0], unit)
    int_c1 = Circle([0.65, 0.65], 0.15, [0, 1, 0], unit)
    int_objects = [int_s1, int_c1]
    for obj in int_objects:
        obj.draw(int_env)
    return int_env, int_ret, int_objects


def external_env_init(unit):
    """Initiate the external environment.

    Keyword arguments:
    - unit -- the size of the sides of the quadratic environment
    """
    ext_env = np.ones([unit, unit, 3])
    ext_ret = Retina([0.5, 0.5], 0.2, [1, 1, 1], unit)
    ext_s1 = Square([0.35, 0.65], 0.15, [1, 0, 0], unit)
    ext_c1 = Circle([0.65, 0.35], 0.15, [0, 1, 0], unit)
    ext_objects = [ext_s1, ext_c1]
    for obj in ext_objects:
        obj.draw(ext_env)
    return ext_env, ext_ret, ext_objects


def redraw_environment(environment, unit, objects):
    """Redraw an environment image.

    Keyword arguments:
    - environment -- the image array of the environment
    - unit -- the size of the sides of the quadratic environment
    - objects -- a list containing the objects in the environment
    """
    environment.fill(1)
    for obj in objects:
        obj.draw(environment)
    return environment


def graphics(int_env, int_objects, int_ret, ext_env, ext_objects, ext_ret,
             unit):
    """Provisory function for plotting the graphics of the system.

    Keyword arguments:
    - int_env -- the image array of the internal environment
    - int_objects -- a list containing the objects in the internal
      environment
    - int_ret -- the retina object in the internal environment
    - ext_env -- the image array of the external environment
    - ext_objects -- a list containing the objects in the external
      environment
    - ext_ret -- the retina object in the external environment
    - unit -- the size of the sides of the quadratic environment
    """
    plt.clf()

    int_env = redraw_environment(int_env, unit, int_objects)
    int_ret_im = int_ret.get_retina_image(int_env)

    plt.subplot(221)
    plt.title('Internal image')
    plt.imshow(int_env)
    # PLOT DESK EDGES
    plt.plot([0.2*unit, 0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit],
             [0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit, 0.2*unit]
             )
    # PLOT RETINA EDGES
    ret_indices = int_ret.get_index_values()
    plt.plot([ret_indices[0][0], ret_indices[0][0], ret_indices[0][1],
              ret_indices[0][1], ret_indices[0][0]],
             [ret_indices[1][0], ret_indices[1][1], ret_indices[1][1],
              ret_indices[1][0], ret_indices[1][0]]
             )

    plt.subplot(222)
    plt.title('Internal retina')
    plt.imshow(int_ret_im)

    ext_env = redraw_environment(ext_env, unit, ext_objects)
    ext_ret_im = ext_ret.get_retina_image(ext_env)

    plt.subplot(223)
    plt.title('External image')
    plt.imshow(ext_env)
    # PLOT DESK EDGES
    plt.plot([0.2*unit, 0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit],
             [0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit, 0.2*unit]
             )
    # PLOT RETINA EDGES
    ret_indices = ext_ret.get_index_values()
    plt.plot([ret_indices[0][0], ret_indices[0][0], ret_indices[0][1],
              ret_indices[0][1], ret_indices[0][0]],
             [ret_indices[1][0], ret_indices[1][1], ret_indices[1][1],
              ret_indices[1][0], ret_indices[1][0]]
             )

    plt.subplot(224)
    plt.title('External retina')
    plt.imshow(ext_ret_im)

    plt.draw()
    plt.pause(0.002)


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
        FUNCTION check_sub_goal() checks if sub_goal_found  # This messes up!
        IF not sub_goal_found or search_step = 0
            FUNCTION foveate(internal_retina) moves internal retina
            FUNCTION check_sub_goal() checks if sub_goal_found
            # MAYBE THIS BELOW SHOULD BE OUTSIDE ANYWAY? JUST CHECK EXTERNAL
            # ENVIRONMENT IF A SUB-GOAL IS FOUND IN INTERNAL ENVIRONMENT?
            SET external_retina position to match internal_retina position
            IF sub_goal_found
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
    number_of_steps = 100
    max_search_steps = 10
    search_step = 0
    image_threshold = 0.003
    sub_goal_found = False
    sub_goal_accomplished = False
    sub_goal_achievable = False
    graphics_on = True

    pixels = 100

    int_env, int_ret, int_objects = internal_env_init(pixels)
    ext_env, ext_ret, ext_objects = external_env_init(pixels)

    # PROVISORY GRAPHICS
    if graphics_on:
        plt.ion()
        plt.figure()
        plt.axis('off')

        graphics(int_env, int_objects, int_ret, ext_env, ext_objects, ext_ret,
                 pixels
                 )

    # MAIN FUNCTIONING
    sub_goal = check_sub_goal(int_ret.center, int_objects)
    if sub_goal:
        sub_goal_found = True
    if sub_goal_found:
        sub_goal_accomplished = goal_accomplished_classifier(
            int_ret.get_retina_image(int_env),
            ext_ret.get_retina_image(ext_env),
            image_threshold
            )

    for step in range(1, number_of_steps+1):
        if search_step >= max_search_steps:
            search_step = 0
            sub_goal_found = False
        if not sub_goal_found:
            foveate(int_ret)
            ext_ret.move(int_ret.center - ext_ret.center)
            sub_goal = check_sub_goal(int_ret.center, int_objects)
            if sub_goal:
                sub_goal_found = True
            if sub_goal_found:
                sub_goal_accomplished = goal_accomplished_classifier(
                    int_ret.get_retina_image(int_env),
                    ext_ret.get_retina_image(ext_env),
                    image_threshold
                    )
        if sub_goal_found and not sub_goal_accomplished:
            search_step += 1
            foveate(ext_ret)
            ext_object = check_sub_goal(ext_ret.center, ext_objects)
            sub_goal_achievable = goal_achievable_classifier(
                int_ret.get_retina_image(int_env),
                ext_ret.get_retina_image(ext_env),
                image_threshold
                )
            if sub_goal_achievable:
                parameterised_skill(ext_ret.center,
                                    int_ret.center,
                                    ext_object)
                ext_env = redraw_environment(ext_env, pixels, ext_objects)
                ext_ret.move(int_ret.center - ext_ret.center)
                sub_goal_accomplished = goal_accomplished_classifier(
                    int_ret.get_retina_image(int_env),
                    ext_ret.get_retina_image(ext_env),
                    image_threshold
                    )
        if sub_goal_accomplished:
            sub_goal_found = False
            sub_goal_accomplished = False
            sub_goal_achievable = False

        if graphics_on:
            graphics(int_env, int_objects, int_ret, ext_env, ext_objects,
                     ext_ret, pixels
                     )


if __name__ == '__main__':
    """
    Here we can run automated tests to check that everything works.

    After we made sure everything works we can just call main() here.

    Make sure to put plots in separate window (%matplotlib qt) to see
    graphics!
    """
    main()

    # Run tests

#    # Set up environment
#    unit = 100
#
#    # INTERNAL
#    int_image = np.ones([unit, unit, 3])
#
#    # Create objects
#    int_c = Circle([0.3, 0.4], 0.15, [1, 0, 0], unit)
#    int_s = Square([0.6, 0.6], 0.1, [0, 0, 1], unit)
#    int_objects = [int_c, int_s]
#
#    int_c.draw(int_image)
#    int_s.draw(int_image)
#
#    plt.figure(1)
#    plt.imshow(int_image)
#
#    import scipy.ndimage as ndimage
#
#    blur = ndimage.gaussian_filter(int_image, sigma=5)
#
#    plt.figure(2)
#    plt.imshow(blur)
