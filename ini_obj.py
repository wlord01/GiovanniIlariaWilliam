#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:30:10 2017

@author: ilaria
"""
import numpy as np
from shapely.geometry import Point as poi

from polygonobjects import Polygon, Circle, Hand, Retina
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_______DATA___INITIALIZATION_____%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#FIGURES - INDICES/LISTS
hand,tri,squ,cir=0,1,2,3
#PLOT
figMov=0 #figure that moves
figVectMov=np.zeros([4,2]) #vector of all figures movement
#DESK
deskCoor=np.array([[1,1],[9,1],[9,9],[1,9]],dtype=float)
objDesk=Polygon(deskCoor)
objDesk.color="none"
#*******************************************************___EXTERNAL_ENVIRONMENT
ext_triCoor=np.array([[1.5,6.5],[3,6.5],[2.25,8]])
ext_squCoor=np.array([[4,6.5],[5.5,6.5],[5.5,8],[4,8]])
ext_cirCoor=np.array([7.2,7.2])
cirRad=0.8
ini_handCoor=np.array([5,1.2])
handRad=0.1
#************************************************************___EXTERNAL_RETINA
ext_pixelsX=11
ext_pixelsY=11
cirFove=0.05
#*******************************************************___INTERNAL_ENVIRONMENT
int_triCoor=np.array([[1.5,6.5],[3,6.5],[2.25,8]])
int_squCoor=np.array([[6,3.5],[7.5,3.5],[7.5,5],[6,5]])
int_cirCoor=np.array([7.2,7.2])
#************************************************************___INTERNAL_RETINA
int_pixelsX=ext_pixelsX
int_pixelsY=ext_pixelsY
cirFove=0.05
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_______OBJ____INITIALIZATION_____%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#******************************************************************************
#**************************************************______EXTERNAL___ENVIRONMENT
#******************************************************************************
#TRIANGLE
ext_objTri=Polygon(ext_triCoor)
ext_objTri.color="g"
#SQUARE
ext_objSqu=Polygon(ext_squCoor)
ext_objSqu.color="b"
#CIRCLE
ext_objCir=Circle(ext_cirCoor,cirRad)
ext_objCir.color="r"
#HAND
handCoor=ini_handCoor
action=[]
objHand=Hand(ini_handCoor,handRad)
objHand.color="y"

ext_figPosList=[[ini_handCoor],[ext_triCoor],[ext_squCoor],[ext_cirCoor]] #list of all figures positions

#******************************************************************************
#*******************************************************______EXTERNAL___RETINA
#******************************************************************************
#______________________________RETINA
ext_retiCoor=deskCoor
ini_ext_retiCoor=deskCoor
ext_retiCoorList=[]
ext_retiCoorList.append(ext_retiCoor)

#matrix of x(trial) * y(vector of pixels)
ext_retiPixALL=np.matrix(np.zeros(ext_pixelsX*ext_pixelsY))
#matrix of x*y pixels of actual trial (es. matrix 11x11 of trial n)
ext_retiPixMat=np.matrix(np.zeros([ext_pixelsX,ext_pixelsY]))
#vector of m pixels of actual trial (es. vector 121 of trial n)
ext_retiPixVec=ext_retiPixALL[0,:]

ext_objReti=Retina(ext_retiCoor,ext_pixelsX,ext_pixelsY)
ext_objReti.linestyle=":"
ext_objReti.edge="k"
#_______________________extract the PIXELS GRID
ext_pixelGrid=ext_objReti.pixelCalc() #calculates list of all pixels coordinates

#this is the definition of the circle to allow the use of "shapely" library
#(to check if pixel polygon intersecates the circle polygon)
#NO need to define it for the other polygons (triangle, square)
#-----> see below 
ext_circle=poi(ext_figPosList[cir][0]).buffer(cirRad) 

Retina.pixelLoop(ext_pixelGrid,ext_objTri.xy,ext_objSqu.xy,ext_circle,ext_retiPixVec)               
ext_retiPixMat=np.flipud(np.reshape(ext_retiPixVec, (ext_pixelsX,ext_pixelsY))) #   MATRIX 
                                                            #   of pixels color
                                                            #np.flipud is necessary to not have the matrix reversed ;)

#_______________________________FOVEA
ext_foveaCoor=ext_objReti.calcFovea() #calculates fovea xy
ext_objFovea=Circle(ext_foveaCoor, cirFove)
ext_foveaList=[ext_foveaCoor] #list of all fovea's random positions

moveFove=np.zeros(2)

#******************************************************************************
#**************************************************______INTERNAL___ENVIRONMENT
#******************************************************************************
#TRIANGLE
int_objTri=Polygon(int_triCoor)
int_objTri.color="g"
#SQUARE
int_objSqu=Polygon(int_squCoor)
int_objSqu.color="b"
#CIRCLE
int_objCir=Circle(int_cirCoor,cirRad)
int_objCir.color="r"

int_figPosList=[[int_triCoor],[int_squCoor],[int_cirCoor]] #list of all figures positions

#******************************************************************************
#*******************************************************______INTERNAL___RETINA
#******************************************************************************
#______________________________RETINA
int_retiCoor=deskCoor
int_retiCoorList=[]
int_retiCoorList.append(int_retiCoor)

#matrix of x(trial) * y(vector of pixels)
int_retiPixALL=np.matrix(np.zeros(int_pixelsX*int_pixelsY))
#matrix of x*y pixels of actual trial (es. matrix 11x11 of trial n)
int_retiPixMat=np.matrix(np.zeros([int_pixelsX,int_pixelsY]))
#vector of x*y pixels of actual trial (es. vector 121 of trial n)
int_retiPixVec=int_retiPixALL[0,:]

int_objReti=Retina(int_retiCoor,int_pixelsX,int_pixelsY)
int_objReti.linestyle=":"
int_objReti.edge="k"

#_______________________extract the PIXELS GRID
int_pixelGrid=ext_pixelGrid #or int_objReti.pixelCalc() #calculates list of all pixels coordinates

#this is the definition of the circle to allow the use of "shapely" library
#(to check if pixel polygon intersecates the circle polygon)
#NO need to define it for the other polygons (triangle, square)
#-----> see below 
int_circle=poi(int_objCir.xy).buffer(cirRad) 

Retina.pixelLoop(int_pixelGrid,int_objTri.xy,int_objSqu.xy,int_circle,int_retiPixVec)
           
int_retiPixMat=np.flipud(np.reshape(int_retiPixVec, (int_pixelsX,int_pixelsY))) #   MATRIX 
                                                        #   of pixels color
                                                        #np.flipud is necessary to not have the matrix reversed ;)

#_______________________________FOVEA

int_foveaCoor=int_objReti.calcFovea() #calculates fovea xy
int_objFovea=Circle(int_foveaCoor, cirFove)

int_foveaList=[int_foveaCoor] #list of all fovea's random positions
