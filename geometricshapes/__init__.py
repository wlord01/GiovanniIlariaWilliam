#!/usr/bin/env python

"""A collection of geometrical objects.

Function:
math_round(x)

Classes:
Shape
Square
Circle
"""


import numpy as np


def math_round(x):
    """Round to integer

    Round arithmetically to nearest integer, with ties going away from
    zero.
    """
    if abs(x) < 1:
        if x >= 0.5:
            x = 1
        else:
            x = 0
    else:
        if x % int(x) >= 0.5:
            x = int(x) + 1
        else:
            x = int(x)
    return x


class Shape(object):
    """Geometric shape.

    Variables:
    - type_ -- type of object (string)
    - center -- center coordinates (floats)
    - size -- float value of object size
    - color -- color as integer (0/1) or RGB list ([R, G, B])
    - unit -- integer unit measure (number of pixels in environment)

    Methods:
    - move -- Move object
    """
    type_ = "Shape"

    def __init__(self, center, size, color, unit):
        self.center = np.array(center)
        self.size = size
        self.color = color
        self.unit = unit

    def move(self, vector):
        """Move object by adding vector to its center position.

        Keyword arguments:
        vector -- 2D vector array

        The method adds the vector to the object's center coordinates
        and trims if the object ends up outside the table
        (0.2 < x, y < 0.8). Trimming is done by checking first if x or
        y is outside the limits. Then check the two functions x = y and
        x = 1 - y to see if the object is over or under these. This
        determines on which coordinate to base the trimming.
        """
        self.center += vector

        # Trim if X or Y is outside table {0.2, 0.8}
        low_limit_ = 0.2
        high_limit_ = 0.8
        center_ = 0.5

        if (not low_limit_ <= self.center[0] <= high_limit_ or
                not low_limit_ <= self.center[1] <= high_limit_):
            # Trim based on which coordinate is farther out
            if (self.center[0] < self.center[1] and
                    self.center[0] > 1 - self.center[1]):
                # Trim based on y
                _y_trim = -abs(self.center[1]-high_limit_)
                _x_trim = -vector[0]*abs(_y_trim)/abs(vector[1])
            elif (self.center[0] > self.center[1] and
                  self.center[0] < 1 - self.center[1]
                  ):
                # Trim based on y
                _y_trim = abs(self.center[1]-low_limit_)
                _x_trim = -vector[0]*abs(_y_trim)/abs(vector[1])
            elif (self.center[0] < self.center[1] and
                  self.center[0] < 1 - self.center[1]
                  ):
                # Trim based on x
                _x_trim = abs(self.center[0]-low_limit_)
                _y_trim = -vector[1]*abs(_x_trim)/abs(vector[0])
            elif (self.center[0] > self.center[1] and
                  self.center[0] > 1 - self.center[1]
                  ):
                # Trim based on x
                _x_trim = -abs(self.center[0]-high_limit_)
                _y_trim = -vector[1]*abs(_x_trim)/abs(vector[0])
            else:
                if self.center[0] > center_:
                    _x_trim = -abs(self.center[0]-high_limit_)
                elif self.center[0] < center_:
                    _x_trim = abs(self.center[0]-low_limit_)
                if self.center[1] > center_:
                    _y_trim = -abs(self.center[1]-high_limit_)
                elif self.center[1] < center_:
                    _y_trim = abs(self.center[1]-low_limit_)

            _trim = np.array([_x_trim, _y_trim])

            self.center += _trim


class Square(Shape):
    """Square object with inheritance from Shape class.

    Added in Square:
    - Methods draw and is_inside

    Variables:
    - type_ -- type of object (string)
    - center -- center coordinates (floats)
    - size -- float value of object size
    - color -- color as integer (0/1) or RGB list ([R, G, B])
    - unit -- integer unit measure (number of pixels in environment)

    Methods:
    - move -- Move object
    - draw -- Draw in image array
    - is_inside -- Tell if point is inside square
    """
    type_ = "Square"

    def get_corners(self):
        """Get the coordinates of the square's corners."""
        _x_min = (self.center[0] - self.size/2)
        _x_max = (self.center[0] + self.size/2)
        _y_min = (self.center[1] - self.size/2)
        _y_max = (self.center[1] + self.size/2)
        return np.array([[_x_min, _x_max], [_y_min, _y_max]])

    def get_index_values(self):
        """Get the coodrinates of the square's corners in index values."""
        _corners = self.get_corners()*self.unit
        _corner_index_values = np.array([[math_round(_corners[0][0]),
                                          math_round(_corners[0][1])],
                                         [math_round(_corners[1][0]),
                                          math_round(_corners[1][1])]
                                         ], dtype=int
                                        )
        return _corner_index_values

    def draw(self, image_array):
        """Draw object in image array.

        Keyword arguments:
        image_array -- the image array to draw in.

        Get coordinates of square's corners, convert to array index
        values and update image_array by coloring the pixels within
        the square.
        """
        _corner_index_values = self.get_index_values()
        image_array[_corner_index_values[1][0]:_corner_index_values[1][1],
                    _corner_index_values[0][0]:_corner_index_values[0][1]
                    ] = self.color

    def is_inside(self, point):
        """Check if point is inside square.

        Keyword arguments:
        point -- array of float coordinates of the point

        If x-coordinate of point is between x_min and x_max of square
        and y-coordinate of point is between y_min and _ymax of square,
        return True. Otherwise return False.
        """
        _corners = self.get_corners()
        if (_corners[0][0] <= point[0] <= _corners[0][1] and
                _corners[1][0] <= point[1] <= _corners[1][1]):
            return True
        else:
            return False


class Retina(Square):
    """Retina class with inheritance from Square.

    Added in Retina:
    - Method get_retina_image(environment_image)

    Variables:
    - type_ -- type of object (string)
    - center -- center coordinates (floats)
    - size -- float value of object size
    - color -- color as integer (0/1) or RGB list ([R, G, B])
    - unit -- integer unit measure

    Methods:
    - move -- Move object
    - draw -- Draw in image array
    - get_retina_image -- Get the array of pixels in the retina
    """
    type_ = "Retina"

    def get_retina_image(self, environment):
        """Get the retina image pixel array.

        Keyword arguments:
        - environment -- the pixel array of the environment the retina
          is in

        Calculate coordinates of retina corners in the environment
        and return array of pixels of the retina.
        """
        _corner_index_values = self.get_index_values()
        _ret_image = environment[
            _corner_index_values[0][0]:_corner_index_values[0][1],
            _corner_index_values[1][0]:_corner_index_values[1][1]
            ]
        return _ret_image


class Circle(Shape):
    """Circle class with inheritance from Shape.

    Added in Circle:
    - self.radius (float)
    - Methods draw and is_inside

    Variables:
    - type_ -- type of object (string)
    - center -- center coordinates (floats)
    - size -- float value of object size
    - color -- color as integer (0/1) or RGB list ([R, G, B])
    - unit -- integer unit measure (number of pixels in environment)
    - radius -- float value of circle radius

    Methods:
    - move -- Move object
    - draw -- Draw in image array
    - is_inside -- Tell if point is inside circle
    """
    type_ = "Circle"

    def __init__(self, center, size, color, unit):
        super(Circle, self).__init__(center, size, color, unit)
        self.radius = self.size/2

    def draw(self, image_array):
        """Draw circle object in image array.

        Keyword arguments:
        image_array -- the image array to draw in

        Takes center (float coordinates), size (float) and image_array
        (matrix array). Updates the array elements in image_array
        which are within radius distance from the center of the circle
        to the color of the circle.
        """
        _x = self.center[0]*self.unit
        _y = self.center[1]*self.unit

        _r = self.radius*self.unit

        _X, _Y = np.meshgrid(np.arange(image_array.shape[0]),
                             np.arange(image_array.shape[1])
                             )

        _d = np.sqrt((_X-_x)**2 + (_Y-_y)**2)

        image_array[np.where(_d < _r)] = self.color

    def is_inside(self, point):
        """Check if point is inside circle object.

        Keyword arguments:
        point -- array of float coordinates of the point

        Takes the center (float coordinates), radius (float) and point
        (float coordinates) and calculates the distance _d between the
        point and the center of the circle. If _d is smaller than the
        radius _r of the circle the point is inside.
        """
        _d = np.sqrt((point[0]-self.center[0])**2
                     + (point[1]-self.center[1])**2
                     )

        if _d <= self.radius:
            return True
        else:
            return False


if __name__ == '__main__':
    # RUN TESTS
    import matplotlib.pyplot as plt

    unit = 100
    image = np.ones([unit, unit, 3])

    s = Square([0.5, 0.5], 0.15, [1, 0, 0], unit)
    s.draw(image)

    retina = Retina([0.5, 0.5], 0.3, [1, 1, 1], unit)
    ret_image = retina.get_retina_image(image)

    plt.imshow(image)
    plt.figure()
    plt.imshow(ret_image)

    retina.move(np.array([0.1, 0.1]))
    ret_image = retina.get_retina_image(image)
    plt.figure()
    plt.imshow(ret_image)

#    import matplotlib.pyplot as plt
#    import matplotlib.patches as patches

    # ALTERNATIVE WAY OF PLOTTING: USE PATCHES
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111, aspect='equal')
    # plt.xlim(0, 1)
    # plt.ylim(1, 0)
    # ax1.add_patch(patches.Rectangle((0.2, 0.2), 0.6, 0.6, fill=False))
    # ax1.add_patch(patches.Rectangle(center, x_size, y_size, color=color))
    # ax1.add_patch(patches.Circle(center, radius, color=color))
    # ax1.plot(point[0], point[1], 'wo')

    pass
