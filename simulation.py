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
import scipy.ndimage as ndimage

from geometricshapes import Square, Circle, Fovea
import actions


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


def check_images(image_array_1, image_array_2, threshold=0.01):
    """Compare two image arrays and say if they show the same thing.

    Takes two numpy arrays and calculates the normalised distance
    between their flattened versions. The threshold is arbitrarily
    chosen and set, to determine if the flattened vectors are similar
    enough.
    """
    vector_1 = image_array_1.flatten('F')
    vector_2 = image_array_2.flatten('F')
    diff = vector_1 - vector_2
    norm = np.linalg.norm(diff)
    if norm/len(vector_1) <= threshold:
        return True
    else:
        return False


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
    same_images = check_images(internal_fovea_image,
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
    same_images = check_images(internal_fovea_image,
                               external_fovea_image,
                               threshold
                               )

    if same_images:
        return True
    else:
        return False


def get_intensity_image(image):
    """
    Translate RGB image to intensity image.

    Keyword arguments:
    - image -- numpy array of RGB image

    Takes the RGB image array, transforms it to black/white image,
    adds Gaussian noise, adds noise and returns this new image.
    """
    bw_image = np.mean(image, -1)
    blurred = ndimage.gaussian_filter(bw_image, sigma=1)
    noisy = blurred + (0.1*np.random.random_sample(blurred.shape) - 0.05)
    clipped = noisy.clip(0, 1)
    return clipped


def foveate(fovea, image):
    """
    Foveate fovea.

    Keyword arguments:
    - fovea -- Fovea object
    - image -- Numpy array of the image the fovea is scanning

    Uses bottom-up attention (the {RGB --> Black/White --> Add noise
    --> (smooth) --> foveate} procedure). That is, RGB image is turned
    into an intensity image, then the fovea is moved to the coordinates
    of the most salient pixel in the image.
    """
    intensity_image = get_intensity_image(image)
    max_index = np.unravel_index(intensity_image.argmax(),
                                 intensity_image.shape
                                 )
    max_pos = np.flipud(np.array(max_index))/image.shape[0]
    fovea.move(max_pos - fovea.center)


def hard_foveate(fovea, image, objects):
    """
    Hard foveation of fovea.

    Keyword arguments:
    - fovea -- Fovea object
    - image -- Numpy array of the image the fovea is scanning
    - objects -- List of objects in the image

    This one uses the bottom-up attention, but then checks which object
    is found, gets its center coordinates and foveates the fovea
    to those coordinates.
    """
    foveate(fovea, image)
    found_object = check_sub_goal(fovea.center, objects)
    fovea.move(found_object.center - fovea.center)


def initialize_environment(unit, fovea_center, fovea_size, objects):
    """Initialize environment

    Keyword arguments:
    - unit -- the size (int) of the sides of the quadratic environment
    - fovea_center -- center coordinates (floats tuple) of fovea
    - fovea_size -- size (float) of fovea
    - objects -- list of objects

    Creates enviroment image array and draws objects in it. Returns
    enviroment image, fovea image and list of objects.

    The environment is RGB color coded pixels. That is each pixel has
    three dimensions.
    """
    environment = np.zeros([unit, unit, 3])
    fovea = Fovea(fovea_center, fovea_size, [0, 0, 0], unit)
    environment = redraw_environment(environment, unit, objects)
    return environment, fovea, objects


def redraw_environment(environment, unit, objects):
    """Redraw an environment image.

    Keyword arguments:
    - environment -- the image array of the environment
    - unit -- the size of the sides of the quadratic environment
    - objects -- a list containing the objects in the environment
    """
    environment.fill(0)
    for obj in objects:
        if obj.center.all():
            obj.draw(environment)
    return environment


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

    int_env = redraw_environment(int_env, unit, int_objects)
    int_fov_im = int_fov.get_focus_image(int_env)

    plt.subplot(221)
    plt.title('Internal image')
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

    ext_env = redraw_environment(ext_env, unit, ext_objects)
    ext_fov_im = ext_fov.get_focus_image(ext_env)

    plt.subplot(223)
    plt.title('External image')
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
            FUNCTION foveate(internal_fovea) moves internal fovea
            FUNCTION check_sub_goal() checks if sub_goal_found
            # MAYBE THIS BELOW SHOULD BE OUTSIDE ANYWAY? JUST CHECK EXTERNAL
            # ENVIRONMENT IF A SUB-GOAL IS FOUND IN INTERNAL ENVIRONMENT?
            SET external_fovea position to match internal_fovea position
            IF sub_goal_found
                FUNCTION goal_accomplished_classifier() checks if
                    sub_goal_accomplished
        IF sub_goal_found and not sub_goal_accomplished
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
            SET sub_goal_found = False
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
    sub_goal_found = False
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

    int_env, int_fov, int_objects = initialize_environment(unit, fovea_center,
                                                           fovea_size,
                                                           int_objects
                                                           )

    # EXTERNAL ENVIRONMENT
    ext_s1 = Square([0.35, 0.65], 0.15, [1, 0, 0], unit)
    ext_c1 = Circle([0.65, 0.35], 0.15, [0, 1, 0], unit)
    ext_objects = [ext_s1, ext_c1]

    ext_env, ext_fov, ext_objects = initialize_environment(unit, fovea_center,
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
    sub_goal = check_sub_goal(int_fov.center, int_objects)
    if sub_goal:
        sub_goal_found = True
    if sub_goal_found:
        sub_goal_accomplished = goal_accomplished_classifier(
            int_fov.get_focus_image(int_env),
            ext_fov.get_focus_image(ext_env),
            accomplished_threshold
            )

    for step in range(1, number_of_steps+1):
        if search_step >= max_search_steps:
            search_step = 0
            sub_goal_found = False
        if not sub_goal_found:
            foveate(int_fov, int_env)
#            hard_foveate(int_fov, int_env, int_objects)
            ext_fov.move(int_fov.center - ext_fov.center)
            sub_goal = True  # check_sub_goal(int_fov.center, int_objects)
            if sub_goal:
                sub_goal_found = True
            if sub_goal_found:
                sub_goal_accomplished = goal_accomplished_classifier(
                    int_fov.get_focus_image(int_env),
                    ext_fov.get_focus_image(ext_env),
                    accomplished_threshold
                    )
        if sub_goal_found and not sub_goal_accomplished:
            search_step += 1
            foveate(ext_fov, ext_env)
#            hard_foveate(ext_fov, ext_env, ext_objects)
            ext_object = check_sub_goal(ext_fov.center, ext_objects)
            sub_goal_achievable = goal_achievable_classifier(
                int_fov.get_focus_image(int_env),
                ext_fov.get_focus_image(ext_env),
                achievable_threshold
                )
            if sub_goal_achievable:
                actions.parameterised_skill(ext_fov.center,
                                            int_fov.center,
                                            ext_object,
                                            limits
                                            )
                ext_env = redraw_environment(ext_env, unit, ext_objects)
                ext_fov.move(int_fov.center - ext_fov.center)
                sub_goal_accomplished = goal_accomplished_classifier(
                    int_fov.get_focus_image(int_env),
                    ext_fov.get_focus_image(ext_env),
                    accomplished_threshold
                    )
        if sub_goal_accomplished:
            sub_goal_found = False
            sub_goal_accomplished = False
            sub_goal_achievable = False
            search_step = 0

        if graphics_on:
            graphics(int_env, int_objects, int_fov, ext_env, ext_objects,
                     ext_fov, unit
                     )

        # BREAK IF GOAL IMAGE IS ACCOMPLISHED
        if check_images(int_env, ext_env, 0.00055):
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
