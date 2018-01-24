# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:02:22 2018

@author: William
"""


def effect_predictors(where_effect_predictors, what_effect_predictors, unit,
                      fovea_size):
    # CHECK EFFECT PREDICTORS
    import numpy as np
    import matplotlib.pyplot as plt
    import environment

    object_images = environment.get_object_images(unit, fovea_size)
    what_effect_predictor = what_effect_predictors[1]
    where_effect_predictor = where_effect_predictors[1]
    for k in range(len(where_effect_predictors)):
        where_effect_predictor = where_effect_predictors[k]
        what_effect_predictor = what_effect_predictors[k]
        for i in object_images:
            image = i[2:]
            what_input = np.array([image]).T
#            end_position = np.array([0.2, 0.8])
            if k == 0:
                where_input = np.array([[0.2, 0.8, 0.5, 0.5]]).T
            else:
                where_input = np.array([[0.5, 0.5]]).T
            where_effect_predictor.set_input(where_input)
            what_effect_predictor.set_input(what_input)
            where_out = where_effect_predictor.get_output()
            what_out = what_effect_predictor.get_output()
            plt.figure()
            plt.subplot(121)
            plt.title(str(where_input))
            pixels = int(fovea_size * unit)
            plt.imshow(np.reshape(what_input, (pixels, pixels, 3), 'F'))
            plt.subplot(122)
            plt.title(str(where_out))
            plt.imshow(np.reshape(what_out, (pixels, pixels, 3), 'F'))