# -*- coding: utf-8 -*-
"""
Environment

Functions:
initialize(unit, fovea_center, fovea_size, objects) - initialize unit size
    environment containing objects and fovea with size fovea_size and center in
    fovea_center.
redraw(environment, unit, objects) - redraw the unit size environment
    containing objects.
"""

import numpy as np

from geometricshapes import Fovea, Square, Circle, Rectangle


def initialize(unit, fovea_center, fovea_size, objects):
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
    environment = redraw(environment, unit, objects)
    return environment, fovea, objects


def redraw(environment, unit, objects):
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


def get_object_images(unit, fovea_size, object_size):
    """Get images of all object shape/color combinations

    Keyword arguments:
    - unit -- the size of the sides of the quadratic environment
    - fovea_size -- size (float) of fovea

    Generates focus images for all possible combinations of object
    shape and color. Returns a list containing these images.
    """
    env = np.zeros([unit, unit, 3])
    fov = Fovea([0.35, 0.65], fovea_size, [1, 1, 1], unit)

    s1 = Square([0.2, 0.2], object_size, [1, 0, 0], unit)
    s2 = Square([0.2, 0.5], object_size, [0, 1, 0], unit)
    s3 = Square([0.2, 0.8], object_size, [0, 0, 1], unit)
    c1 = Circle([0.5, 0.2], object_size, [1, 0, 0], unit)
    c2 = Circle([0.5, 0.5], object_size, [0, 1, 0], unit)
    c3 = Circle([0.5, 0.8], object_size, [0, 0, 1], unit)
    b1 = Rectangle([0.8, 0.2], object_size, [1, 0, 0], unit, 0)
    b2 = Rectangle([0.8, 0.5], object_size, [0, 1, 0], unit, 0)
    b3 = Rectangle([0.8, 0.8], object_size, [0, 0, 1], unit, 0)

    object_images = []
    objects = [s1, s2, s3, c1, c2, c3, b1, b2, b3]
    for obj in objects:
        obj.draw(env)
        center = obj.center
        fov.center = np.array(center)
        fov_im = fov.get_focus_image(env)

        if obj.type_ == 'Square':
            object_type = 0
        elif obj.type_ == 'Circle':
            object_type = 1
        elif obj.type_ == 'Rectangle':
            object_type = 2

        object_color = obj.color.index(1)

        object_image = np.concatenate((np.array([object_type, object_color]),
                                       fov_im.flatten('F').T))
        object_images.append(object_image)

#    centers = ([0.2, 0.2], [0.2, 0.5], [0.2, 0.8], [0.5, 0.2], [0.5, 0.5],
#               [0.5, 0.8], [0.8, 0.2], [0.8, 0.5], [0.8, 0.8])
#
#
#    for center in centers:
#        fov.center = np.array(center)
#        fov_im = fov.get_focus_image(env)
#        object_images.append(fov_im)

    return object_images


if __name__ == '__main__':
    """Main"""
    import matplotlib.pyplot as plt
#    object_images = get_object_images(150, 0.14, 0.1)
#    focus_image_pixels = int(0.14*150)
#    for image in object_images:
#        plt.figure()
#        image = np.reshape(image[2:], (focus_image_pixels, focus_image_pixels,
#                           3), 'F'
#                           )
#        plt.imshow(image)

#   #### PLOT ENVIRONMENT AND GOAL IMAGE ####
    unit = 150
    object_size = 0.1
    fovea_center = [0.5, 0.5]
    fovea_size = 0.14

    # INITIALIZE INTERNAL ENVIRONMENT
    int_s1 = Square([0.2, 0.5], object_size, [0, 1, 0], unit, 3)
    int_r1 = Rectangle([0.8, 0.2], object_size, [1, 0, 0], unit, 0, 5)
    int_s2 = Square([0.2, 0.2], object_size, [1, 0, 0], unit, 10)
    int_c1 = Circle([0.5, 0.8], object_size, [0, 0, 1], unit, 3)
    int_c2 = Circle([0.5, 0.2], object_size, [1, 0, 0], unit, 8)
    int_objects = [int_s1, int_r1, int_s2, int_c1, int_c2]

    int_env, int_fov, int_objects = initialize(unit, fovea_center, fovea_size,
                                               int_objects
                                               )

    # INITIALIZE EXTERNAL ENVIRONMENT
    ext_s1 = Square([0.2, 0.5], object_size, [0, 0, 1], unit)
    ext_r1 = Rectangle([0.8, 0.2], object_size, [0, 1, 0], unit, 0)
    ext_s2 = Square([0.2, 0.8], object_size, [1, 0, 0], unit)
    ext_c1 = Circle([0.5, 0.8], object_size, [0, 1, 0], unit)
    ext_c2 = Circle([0.5, 0.5], object_size, [1, 0, 0], unit)
    ext_objects = [ext_s1, ext_r1, ext_s2, ext_c1, ext_c2]

    ext_env, ext_fov, ext_objects = initialize(unit, fovea_center, fovea_size,
                                               ext_objects
                                               )

    plt.figure()
    plt.subplot(121)
    plt.title('Goal image')
    plt.xlim(0, unit)
    plt.ylim(0, unit)
    plt.axis('off')
    plt.imshow(int_env, origin='lower')
    # PLOT DESK EDGES
    plt.plot([0.2*unit, 0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit],
             [0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit, 0.2*unit], 'w-'
             )
    plt.subplot(122)
    plt.title('Environment')
    plt.xlim(0, unit)
    plt.ylim(0, unit)
    plt.axis('off')
    plt.imshow(ext_env, origin='lower')
    # PLOT DESK EDGES
    plt.plot([0.2*unit, 0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit],
             [0.2*unit, 0.8*unit, 0.8*unit, 0.2*unit, 0.2*unit], 'w-'
             )
