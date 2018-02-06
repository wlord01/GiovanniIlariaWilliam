#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perception of the system.

Functions:
- check_sub_goal() -- Check if current position of fovea is on a sub-goal
- get_intensity_image() -- Translate RGB image into intensity image
- foveate() -- Foveate fovea
- hard_foveate() -- Hard foveation of fovea to center of object
- check_effect() -- Check effect of action
"""

import numpy as np
import scipy.ndimage as ndimage


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


def foveate(fovea, image, objects):
    """
    Foveate fovea.

    Keyword arguments:
    - fovea -- Fovea object
    - image -- Numpy array of the image the fovea is scanning
    - objects -- List of objects in the image

    Uses bottom-up attention (the {RGB --> Black/White --> Add noise
    --> (smooth) --> foveate} procedure). That is, RGB image is turned
    into an intensity image, then the fovea is moved to the coordinates
    of the most salient pixel in the image.

    The if- and while-loops make sure the system foveates to a new
    object if there are more than one object in the environment.
    """
    current_object = check_sub_goal(fovea.center, objects)

    intensity_image = get_intensity_image(image)
    max_index = np.unravel_index(intensity_image.argmax(),
                                 intensity_image.shape
                                 )
    max_pos = np.flipud(np.array(max_index))/image.shape[0]
    fovea.move(max_pos - fovea.center)

    if len(objects) >> 1:
        new_object = check_sub_goal(fovea.center, objects)

        while new_object == current_object:
            intensity_image = get_intensity_image(image)
            max_index = np.unravel_index(intensity_image.argmax(),
                                         intensity_image.shape
                                         )
            max_pos = np.flipud(np.array(max_index))/image.shape[0]
            fovea.move(max_pos - fovea.center)
            new_object = check_sub_goal(fovea.center, objects)


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
    foveate(fovea, image, objects)
    found_object = check_sub_goal(fovea.center, objects)
    fovea.move(found_object.center - fovea.center)


def effect_foveate(fovea, image, objects):
    """
    Foveation on effect image

    Keyword arguments:
    - fovea -- Fovea object
    - image -- Effect image numpy array
    - objects -- List of objects in environment

    This works like hard_foveate() but without the requirement of
    moving to a new object, since the effect is always on one object
    only.
    """
    intensity_image = get_intensity_image(image)
    max_index = np.unravel_index(intensity_image.argmax(),
                                 intensity_image.shape
                                 )
    max_pos = np.flipud(np.array(max_index))/image.shape[0]
    found_object = check_sub_goal(max_pos, objects)
    fovea.move(found_object.center - fovea.center)


def check_effect(before_image, after_image):
    """Check effect of action

    Keyword arguments:
    - before_image -- image array of enviroment before action.
    - after_image -- image array of environment after action.

    Checks the difference image (after - before) for any change. If the change
    is above a threshold, the system notes an effect. If the system notes an
    effect if foveates to the positive pixels (things added after action) and
    returns this state. If change is below threshold False is returned.
    """
    difference_image = after_image - before_image

    if check_images(difference_image, np.zeros(difference_image.shape), 1E-5):
        # IF DIFFERENCE IMAGE IS ALL BLACK THERE WAS NO EFFECT
        return False
    else:
        # THERE WAS AN EFFECT
        return True


if __name__ == '__main__':
    """Main"""
