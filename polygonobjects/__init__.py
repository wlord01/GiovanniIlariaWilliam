#!/usr/bin/env python

"""A collection of polygon objects created with shapely library"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc


class Polygon(object):
    """Polygon"""
    def __init__(self,xy): #CONSTRUCTOR
        self.xy=xy
        self.color="none"
        self.linestyle="-"
        self.edge="k"
        
    def randPos(self): #random position in x/y limits
        randomX=np.random.uniform(0,10)
        randomY=np.random.uniform(0,10)
        randomXY=np.array([randomX,randomY])
        return randomXY
        
    def plotDef(self): #define the plot
        self.figure=ptc.Polygon(self.xy,facecolor = self.color, 
                                linestyle=self.linestyle, edgecolor=self.edge)
                 
    def plotAdd(self,numFig,title): #add plot to figure
        self.xLim=(-5,15)      
        self.yLim=(-5,15)
        self.fig=plt.figure(numFig) 
        self.ax1 = self.fig.add_subplot(111,aspect='equal')       
        self.ax1.set_xlim((self.xLim))
        self.ax1.set_ylim((self.yLim))
        self.ax1.set_title(title)
        
        self.ax1.add_patch(self.figure)
        
    def animateFig(self,coorList,position,numFig,title): #plot animation
        self.xy=coorList[position]
        
        self.figure.remove()
        self.plotDef()
        self.plotAdd(numFig,title)


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