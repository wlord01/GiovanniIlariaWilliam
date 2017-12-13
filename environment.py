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

from geometricshapes import Fovea


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


if __name__ == '__main__':
    """Main"""
