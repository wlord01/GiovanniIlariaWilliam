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

        startpoint = self.center
        endpoint = self.center + vector
        limits = np.array([[0.2, 0.8], [0.2, 0.8]])  # MAKE GENERAL

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

        self.center += vector


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
            _corner_index_values[1][0]:_corner_index_values[1][1],
            _corner_index_values[0][0]:_corner_index_values[0][1]
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
    import matplotlib.patches as patches
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(patches.Rectangle((0.2, 0.2), 0.6, 0.6, fill=False))

    startpoint = np.array([0.6, 0.5])
#    c = Circle(startpoint, 0.1, 1, 10)

    endpoints = np.array([[0.1, 0.5], [0.1, 0.15], [0.1, 0.1], [0.15, 0.1],
                          [0.5, 0.1], [0.85, 0.1], [0.9, 0.1], [0.9, 0.15],
                          [0.9, 0.5], [0.9, 0.85], [0.9, 0.9], [0.85, 0.9],
                          [0.5, 0.9], [0.15, 0.9], [0.1, 0.9], [0.1, 0.85]
                          ]
                         )
#    endpoints = np.array([[0.5, 0.1]])

    c_list = []
    for i in range(len(endpoints)):
        c_list.append(Circle(startpoint, 0.1, 1, 10))

    for i in range(len(endpoints)):
        c = c_list[i]
        endpointz = endpoints[i]

        vectorz = endpointz - c.center
        point2 = startpoint + vectorz

        ax1.plot(c.center[0], c.center[1], 'o')

        c.move(vectorz)
        ax1.plot(c.center[0], c.center[1], 'o')

        #    ax1.plot(point2[0], point2[1], 'o')
        ax1.plot([startpoint[0], endpointz[0]], [startpoint[1], endpointz[1]])

        plt.xlim(-0.2, 1.2)
        plt.ylim(1.2, -0.2)
        plt.show()
        plt.pause(0.002)
        print(c.center)

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
