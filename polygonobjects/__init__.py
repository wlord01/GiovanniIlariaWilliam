#!/usr/bin/env python

"""A collection of geometrical objects.

Function:
math_round(x)

Classes:
Polygon
Square
Circle
Retina
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc


def math_round(x):
    """Round to integer

    Round arithmetically to nearest integer, with ties going away from zero.
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


class Polygon(object):
    """Polygon object

    Variables:
    - Type (string?)
    - Color (integer? RGB code?)
    - Center coordinates (floats)
    - Size (integer? Or float?)
    - Coordinates of object with respect to center

    Methods:
    - Move object
    - Draw on table
    - Tell if point is inside
    """
    type_ = "Polygon"

    def __init__(self, center, size, color, unit):
        self.center = center
        self.size = size
        self.color = color
        self.unit = unit

        # Old stuff
        self.linestyle = "-"
        self.edge = "k"

    def randPos(self):  # Old stuff
        randomX = np.random.uniform(0, 10)
        randomY = np.random.uniform(0, 10)
        randomXY = np.array([randomX, randomY])
        return randomXY

    def plotDef(self):  # Old stuff
        self.figure = ptc.Polygon(self.xy, facecolor=self.color,
                                  linestyle=self.linestyle,
                                  edgecolor=self.edge
                                  )

    def plotAdd(self,numFig,title): # Old stuff
        self.xLim=(-5,15)      
        self.yLim=(-5,15)
        self.fig=plt.figure(numFig) 
        self.ax1 = self.fig.add_subplot(111,aspect='equal')       
        self.ax1.set_xlim((self.xLim))
        self.ax1.set_ylim((self.yLim))
        self.ax1.set_title(title)
        
        self.ax1.add_patch(self.figure)
        
    def animateFig(self,coorList,position,numFig,title): # Old stuff
        self.xy=coorList[position]
        
        self.figure.remove()
        self.plotDef()
        self.plotAdd(numFig,title)


class Square(Polygon):
    """Square object with inheritance from Polygon class.

    Variables:
    - Type (string)
    - Color (integer 0/1 or RGB list)
    - Center coordinates (floats)
    - Size (float)

    Methods:
    - Move object
    - Draw on table
    - Tell if point is inside
    """
    type_ = "Square"

    def move(self, vector):
        """Move object by adding vector to its center position."""
        self.center += vector

    def get_corners(self):
        """Get the corner coordinates of the square."""
        _center = self.center*(self.unit-1)  # Pixel-value
        _size = self.size*self.unit
        _x_min = math_round(_center[0] - _size/2)
        _x_max = math_round(_center[0] + _size/2)
        _y_min = math_round(_center[1] - _size/2)
        _y_max = math_round(_center[1] + _size/2)
        _corner_coordinates = [np.array([_x_min, _x_max], dtype=int),
                               np.array([_y_min, _y_max], dtype=int)
                               ]
        return _corner_coordinates

    def draw(self, grid):
        """Draw object in grid.

        Convert position and size to array index values and update grid.

        Keyword arguments:
        grid -- the grid to draw in
        unit_measure -- the size of the grid
        """
        _corner_coordinates = self.get_corners()
        grid[_corner_coordinates[1][0]:_corner_coordinates[1][1],
             _corner_coordinates[0][0]:_corner_coordinates[0][1]
             ] = self.color

    def is_inside(self, point):
        """Check if point is inside square.

        If x-coordinate of point is between x_min and x_max of square and
        y-coordinate of point is between y_min and _ymax of square, return
        True. Otherwise return False.
        """
        _x_min = self.center[0] - self.size/2
        _x_max = self.center[0] + self.size/2
        _y_min = self.center[1] - self.size/2
        _y_max = self.center[1] + self.size/2
        if _x_min <= point[0] <= _x_max and _y_min <= point[1] <= _y_max:
            return True
        else:
            return False


class Circle(Polygon):
    """Circle"""
    def __init__(self,xy,radius):
        self.xy=xy
        self.radius=radius
        self.color="none"

    def plotDef(self):
        self.figure=ptc.Circle((self.xy),self.radius, 
                            facecolor = self.color)


class Hand(Circle):
    """Hand"""
    def __init__(self,xy,radius):
        Circle.__init__(self,xy,radius)


class Retina(Polygon):
    """Retina"""
    def __init__(self,xy, row_amount,col_amount): #CONSTRUCTOR
        self.xy=xy
        self.color="none"    
        self.col_amount=col_amount #columns of pixels
        self.row_amount=row_amount #rows of pixels
        
        self.widthX=(max(self.xy[:,0])-min(self.xy[:,0]))/self.col_amount #resolution(x)
        self.widthY=(max(self.xy[:,1])-min(self.xy[:,1]))/self.row_amount #resolution(y)
        
    def calcFovea(self): #calculates the center of the polygon retina 
        #VERTICES-from bottom left-ABCD
        x=(max(self.xy[:,0])+min(self.xy[:,0]))/2 #lenght of AB/2
        y=(max(self.xy[:,1])+min(self.xy[:,1]))/2 #lenght of AD/2
        fovea=np.array([x,y]) #fovea coordinates
        return fovea
        
    def plotAdd(self,numFig,title):
        #modified, to plot the grid of retina on the setting
        self.xLim=(-5,15)      
        self.yLim=(-5,15)
        self.fig=plt.figure(numFig) 
        self.ax1 = self.fig.add_subplot(111,aspect='equal')       
        self.ax1.set_xlim((self.xLim))
        self.ax1.set_ylim((self.yLim))
        self.ax1.set_title(title)

        self.ax1.add_patch(self.figure)
        
        #define range of grid (row/col)
        self.ax1.set_xticks(np.arange(min(self.xy[:,0]),max(self.xy[:,0]), self.widthX))
        self.ax1.set_yticks(np.arange(min(self.xy[:,1]), max(self.xy[:,1]), self.widthY))
        self.ax1.grid(linewidth=1,clip_path=self.figure) 
        plt.show()
        
    def pixelCalc(self):#calculates the coordinates for every pixel in the retina
        #from the bottom left one:
        iniX=min(self.xy[:,0])
        iniY=min(self.xy[:,1]) 
        pixel_list = [] # empty list to hold all pixels  
        
        #PIXELS iteration
        for row in range(0,self.row_amount):
            y0=iniY
            y1=y0+self.widthY
            iniY=y1
            iniX=min(self.xy[:,0])
            
            for col in range(0,self.col_amount):
                x0=iniX
                x1=x0+self.widthX
                iniX=x1
                #coordinates of vertices of pixel xy
                pixelCoor=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],dtype=float)
                pixel_list.append(pixelCoor)
                      
        return pixel_list

    def pixelLoop(grid,tri,squ,cir,retiPixVec):
        cell=-1 #pixel
        for pix in (grid): #for every pixel of the retina's grid --->
            cell+=1
            
            #___check if PIXEL is on a FIGURE, then: 
                                              #   add the value RGB in retinaVector
                                              #   depending on which is the figure 
            
            #----->below..
            if (poly(tri).intersects(poly(pix)))==True:
                retiPixVec[0,cell]=green
            elif (poly(squ).intersects(poly(pix)))==True:
                retiPixVec[0,cell]=red
            elif ((cir).intersects(poly(pix))) == True:
                retiPixVec[0,cell]=blue
            else:
                retiPixVec[0,cell]=others #   if pixel doesn't overlapping figures
            #.....below again :P    
            #N.B. (poly .... intersects) is a function of "shapely" library; it allows
            #to check if two figures intersecates in 3 ways:
            #   1. True if two figures edges are in touch
            #   2. True if one figure overlap the other
            #   3. True if one figure is inside the other
            #So it works even if only one point of the figure in on one of the other.
            #*we should consider to use it instead of checkPoint and checkPath! ;)

if __name__ == '__main__':
    # Run tests
    plt.close('all')

    external_size = 1024

    # Put grid as ones for RGB images
#    external_grid = np.ones([external_size, external_size, 3])
    external_grid = np.zeros([external_size, external_size])

    p1 = Square(np.array([0.5, 0.5], dtype=float), 0.2, 1, external_size)

    # Before square is put in
#    plt.imshow(external_grid)
#    plt.show()

    plt.figure()
    print(external_grid)  # Before square is put in
    p1.draw(external_grid)
    plt.imshow(external_grid)
    print(external_grid)  # After square is put in

    # Test points for is_inside()
    point = np.array([0.53, 0.457])
    print(p1.is_inside(point))
    plt_point = point*(external_size-1)
    plt.plot(plt_point[0], plt_point[1], 'wo')
#
#    p1.move(np.array([0.1, 0.1]))
#    p1.draw(external_grid)
#    plt.imshow(external_grid)
#
#    p1.move(np.array([-0.1, 0.1]))
#    p1.draw(external_grid)
#    plt.imshow(external_grid)
    
    plt.show()
