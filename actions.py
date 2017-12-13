#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Actions/affordances of the system.

Functions:
activate(object_) - Changes the object color from red to green.
"""

import numpy as np


def activate(object_):
    """Activate object by making it green

    Keyword arguments:
    - object_ -- The object (class instance) to activate

    Turn object green if object color is red"""
    if object_.color == [1, 0, 0]:
        object_.color = np.array([0, 1, 0])

if __name__ == '__main__':
    """Main"""
