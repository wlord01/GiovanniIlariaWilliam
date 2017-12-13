#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Actions/affordances of the system.

Functions:
activate(object_) - Changes the object color from red to green.
parameterised_skill(end_position, start_position, object_, limits) - Move
    object from start position to end position. Trim movement vector to within
    limits of environment.
"""

import numpy as np
import scipy.ndimage as ndimage


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
            if polygon.is_inside(fovea_coordinates):
                found = True
                return polygon

    if found:
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


def get_line_equation(startpoint, endpoint):
        """Return k and m in the line equation y = k*x + m"""
        return np.polyfit([startpoint[0], endpoint[0]],
                          [startpoint[1], endpoint[1]],
                          1)


def get_x_intersection(x_limit, k, m):
    """Return intersection point of x_limit line and y = k*x + m"""
    return np.array([x_limit, x_limit*k + m])


def get_y_intersection(y_limit, k, m):
    """Return intersection point of y_limit line and y = k*x + m"""
    return np.array([(y_limit-m) / k, y_limit])


def x_based_trim(endpoint, vector, x_limit):
    """Trim vector based on x"""
    _x_trim = -(endpoint[0] - x_limit)
    _y_trim = (_x_trim*vector[1]/vector[0])
    _trim = np.array([_x_trim, _y_trim])
    vector += _trim
    return vector


def y_based_trim(endpoint, vector, y_limit):
    """Trim vector based on y"""
    _y_trim = -(endpoint[1] - y_limit)
    _x_trim = (_y_trim*vector[0]/vector[1])
    _trim = np.array([_x_trim, _y_trim])
    vector += _trim
    return vector


def move_object(object_, vector, limits):
    """Move the objects center along the vector.

    Keyword arguments:
    - object_ -- the object to be moved
    - vector -- numpy array of the 2D vector from starting point to end
    point
    - limits -- numpy array of [[x_min, x_max], [y_min, y_max]] limits

    The method checks the proposed move and trims if the object would
    end up outside the table limits. Trimming is done by checking first
    if x or y is outside the limits. Then check if one of x or y is
    inside the limits. If x is inside, the vector is trimmed based on
    y. If y is inside, the vector is trimmed based on x. If both x and
    y are outside the limits the algorithm checks which limit is
    intersected first along the vector from start point to end point.
    If the x limit is intersected first, the vector is trimmed based
    on x and if the y limit is intersected first, the vector is trimmed
    based on y.
    """
    startpoint = object_.center
    endpoint = object_.center + vector

    if (not limits[0][0] <= endpoint[0] <= limits[0][1] or
            not limits[1][0] <= endpoint[1] <= limits[1][1]):
        # ENDPOINT OUTSIDE LIMITS
        if limits[0][0] <= endpoint[0] <= limits[0][1]:
            # INSIDE X-LIMIT --> TRIM BASED ON Y
            if endpoint[1] < limits[1][0]:
                y_limit = limits[1][0]
            elif endpoint[1] > limits[1][1]:
                y_limit = limits[1][1]

            vector = y_based_trim(endpoint, vector, y_limit)

        elif limits[1][0] <= endpoint[1] <= limits[1][1]:
            # INSIDE Y-LIMIT --> TRIM BASED ON X
            if endpoint[0] < limits[0][0]:
                x_limit = limits[0][0]
            elif endpoint[0] > limits[0][1]:
                x_limit = limits[0][1]

            vector = x_based_trim(endpoint, vector, x_limit)

        else:
            # OUTSIDE BOTH X- AND Y-LIMIT
            k, m = get_line_equation(startpoint, endpoint)

            if endpoint[0] < limits[0][0] and endpoint[1] < limits[1][0]:
                # OUTSIDE LOWER LIMITS
                x_limit = limits[0][0]
                y_limit = limits[1][0]
            elif (endpoint[0] < limits[0][0] and
                    endpoint[1] > limits[1][1]):
                # OUTSIDE LOWER X AND UPPER Y
                x_limit = limits[0][0]
                y_limit = limits[1][1]
            elif (endpoint[0] > limits[0][1] and
                    endpoint[1] > limits[1][1]):
                # OUTSIDE BOTH UPPER LIMITS
                x_limit = limits[0][1]
                y_limit = limits[1][1]
            elif (endpoint[0] > limits[0][1] and
                    endpoint[1] < limits[1][0]):
                # OUTSIDE UPPER X AND LOWER Y
                x_limit = limits[0][1]
                y_limit = limits[1][0]

            if (np.linalg.norm(
                    get_x_intersection(x_limit, k, m) -
                    startpoint
                    ) <=
                np.linalg.norm(
                    get_y_intersection(y_limit, k, m) -
                    startpoint
                    )):
                # INTERSECTS X BEFORE Y
                vector = x_based_trim(endpoint, vector, x_limit)
            else:
                # INTERSECTS Y BEFORE X
                vector = y_based_trim(endpoint, vector, y_limit)

    object_.move(vector)


def parameterised_skill(end_position, start_position, object_, limits):
    """
    Move object in external environment.

    This parameterised skill has to learn. This will be the main part of
    the intelligent system. We will implement this using RBM.

    FOR NOW: Implement hardwired moving of objects.

    Arguments:
    - end_position -- array of float coordinates of end position
    - start_position -- array of float coordinates of start position
    - object_ -- the object that the function should move
    """
    vector = start_position - end_position
    move_object(object_, vector, limits)


def activate(object_):
    """Activate object by making it green

    Keyword arguments:
    - object_ -- The object (class instance) to activate

    Turn object green if object color is red"""
    if object_.color == [1, 0, 0]:
        object_.color = np.array([0, 1, 0])

if __name__ == '__main__':
    """Main"""
