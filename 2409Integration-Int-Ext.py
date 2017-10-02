# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 20:16:21 2017

@author: ilaria
"""

"""
This is how the retina conceptually works:

Retina is represented on the simulator by a GRID of pixels; so the Retina is a
"polygon" composed by nXm "sub-polygons" (pixels) - in this case 11x11 pixels.
All these polygons are also defined by them vertices, to allow the use of functions 
like "intercepts" (from "shapely" library), that works like checkPath/checkPoint.

The movement of the Retina is based on the movement of its "fovea": at each
retina's step corresponds the calculation of random xy of fovea and then the new
retina's vertices are calculated (along with the new pixel's xy vertices).

So, at every saccade it's checked -for every pixel- if the "pixel polygon"
intersecates one of the "figure polygon" (triangle, square, circle); if so, 
it's added a value in the "matrix of eye view", depending on which is the
figure the pixel intersecates (1=tri,2=squ,3=cir); otherwise it's addes a 0 in 
the matrix.

In the end, we'll have a matrix that represents the "external vision", that
we plot with imshow function from matplotlib lybrary.

The image is obviously more confused the less the pixels are (for example, try
with 80x80 pixels, better huh? :P), but for now we use low resolution (11x11).
 
..the code has to be optimized! (see at the end of the code)

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
#from matplotlib.path import Path
from matplotlib import animation
from shapely.geometry import Polygon as poly #for check pixel/figure intersection
from shapely.geometry import Point as poi

from polygonobjects import Polygon, Circle, Hand, Retina

plt.ion()
###############################################################_________METHODS       
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
      
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_______DATA___INITIALIZATION_____%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          
#COLORS for eye plot
green,red,blue,others=1,2,3,0
#FIGURES - INDICES/LISTS
hand,tri,squ,cir=0,1,2,3

figMov=0 #figure that moves
figVectMov=np.zeros([4,2]) #vector of all figures movement
# or -----> figPosMat=np.matrix([handCoor,ext_triCoor,ext_squCoor,ext_cirCoor])

#DESK
deskCoor=np.array([[1,1],[9,1],[9,9],[1,9]],dtype=float)
objDesk=Polygon(deskCoor)
objDesk.color="none"

#*******************************************************___EXTERNAL_ENVIRONMENT
#******************************************************************************
#TRIANGLE
ext_triCoor=np.array([[1.5,6.5],[3,6.5],[2.25,8]])
ext_objTri=Polygon(ext_triCoor)
ext_objTri.color="g"

#SQUARE
ext_squCoor=np.array([[4,6.5],[5.5,6.5],[5.5,8],[4,8]])
ext_objSqu=Polygon(ext_squCoor)
ext_objSqu.color="b"

#CIRCLE
ext_cirCoor=np.array([7.2,7.2])
cirRad=0.8
ext_objCir=Circle(ext_cirCoor,cirRad)
ext_objCir.color="r"

#HAND
ini_handCoor=np.array([5,1.2])
handCoor=np.array([5,1.2])
handRad=0.1
action=[]
#actMat=np.matrix(np.zeros([2,2])) #action matrix (x0,y0;x1,y1)
objHand=Hand(ini_handCoor,handRad)
objHand.color="y"

ext_figPosList=[[ini_handCoor],[ext_triCoor],[ext_squCoor],[ext_cirCoor]] #list of all figures positions

#************************************************************___EXTERNAL_RETINA
#******************************************************************************
#______________________________RETINA
ext_pixelsX=11
ext_pixelsY=11

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

pixelLoop(ext_pixelGrid,ext_objTri.xy,ext_objSqu.xy,ext_circle,ext_retiPixVec)               
ext_retiPixMat=np.flipud(np.reshape(ext_retiPixVec, (ext_pixelsX,ext_pixelsY))) #   MATRIX 
                                                            #   of pixels color
                                                            #np.flipud is necessary to not have the matrix reversed ;)

#_______________________________FOVEA
ext_foveaCoor=ext_objReti.calcFovea() #calculates fovea xy
ext_objFovea=Circle(ext_foveaCoor, 0.05)
ext_foveaList=[ext_foveaCoor] #list of all fovea's random positions

moveFove=np.zeros(2)
#*************************************************___INTERNAL_ENVIRONMENT(GOAL)
#******************************************************************************
#TRIANGLE
int_triCoor=np.array([[1.5,6.5],[3,6.5],[2.25,8]])
int_objTri=Polygon(int_triCoor)
int_objTri.color="g"

#SQUARE
int_squCoor=np.array([[6,3.5],[7.5,3.5],[7.5,5],[6,5]])
int_objSqu=Polygon(int_squCoor)
int_objSqu.color="b"

#CIRCLE
int_cirCoor=np.array([7.2,7.2])
int_objCir=Circle(int_cirCoor,cirRad)
int_objCir.color="r"

int_figPosList=[[int_triCoor],[int_squCoor],[int_cirCoor]] #list of all figures positions

#************************************************************___INTERNAL_RETINA
#******************************************************************************
#______________________________RETINA
int_pixelsX=ext_pixelsX
int_pixelsY=ext_pixelsY

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

pixelLoop(int_pixelGrid,int_objTri.xy,int_objSqu.xy,int_circle,int_retiPixVec)
           
int_retiPixMat=np.flipud(np.reshape(int_retiPixVec, (int_pixelsX,int_pixelsY))) #   MATRIX 
                                                        #   of pixels color
                                                        #np.flipud is necessary to not have the matrix reversed ;)

#_______________________________FOVEA

int_foveaCoor=int_objReti.calcFovea() #calculates fovea xy
int_objFovea=Circle(int_foveaCoor, 0.05)

int_foveaList=[int_foveaCoor] #list of all fovea's random positions


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_____________S_T_A_R_T___________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
graphic="on"
#TRIALS
numTri=0
numMov=0
maxTri=50
while numTri < maxTri and ((poi(objHand.xy).within(poly(ext_objTri.xy))) != True)\
                      and ((poi(objHand.xy).within(poly(ext_objSqu.xy))) != True)\
                      and ((poi(objHand.xy).within(ext_circle)) != True):
                      #MODIFY WITH CHECKING IF EXTERNAL RETINA != INTERNAL RETINA!!!!!!!!!!!!!!!!!!
                          
    numTri+=1
    numMov+=1

    #New random HAND position    
    ext_figPosList[hand].append(objHand.randPos())
    handCoor=ext_figPosList[hand][numTri]
    objHand.xy=handCoor

    #New position set
    ext_figPosList[tri].append(ext_triCoor)
    ext_figPosList[squ].append(ext_squCoor)
    ext_figPosList[cir].append(ext_cirCoor)
     
    
    #New random RETINA position
    ext_foveaList.append(ext_objReti.randPos()) #new random fovea position
    ext_foveaCoor=ext_foveaList[numTri]
    
    int_foveaList.append(ext_foveaList[-1])
    int_foveaCoor=int_foveaList[numTri]
    
    
         #___check if FOVEA is inside the desk
    while True:
        #if NO ---> calculate new fovea position..
        if (poi(ext_foveaCoor).within(poly(objDesk.xy))) != True: 
            ext_foveaList[numTri]=ext_objReti.randPos()
            ext_foveaCoor=ext_foveaList[numTri]
            
            int_foveaList[numTri]=ext_foveaList[numTri]
            int_foveaCoor=int_foveaList[numTri]
        #..until YES! ---> fovea is inside the desk
        else:
            break   
        
        #___MOVE RETINA
    moveFove=ext_foveaList[numTri] - ext_foveaList[numTri-1] #calculate fovea movement
    ext_retiCoor=ext_retiCoor+moveFove.T #calculate new position of retina vertices
    ext_retiCoorList.append(ext_retiCoor)    
    ext_objReti.xy=ext_retiCoor
    
    int_retiCoor=int_retiCoor+moveFove.T
    int_retiCoorList.append(int_retiCoor)
    int_objReti.xy=int_retiCoor
    
#!!!!!!!#___extract the PIXELS GRID_______________@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #as above!
    ext_pixelGrid=ext_objReti.pixelCalc() #calculates list of all pixels new coordinates
    ext_circle=poi(ext_figPosList[cir][numTri]).buffer(cirRad) 
    #insert row for the new trial retina's position
    ext_retiPixALL=np.vstack((ext_retiPixALL,np.matrix(np.zeros(ext_pixelsX*ext_pixelsY))))
    ext_retiPixVec=ext_retiPixALL[numTri,:]

    pixelLoop(ext_pixelGrid,ext_objTri.xy,ext_objSqu.xy,ext_circle,ext_retiPixVec)       
    ext_retiPixMat=np.flipud(np.reshape(ext_retiPixVec, (ext_pixelsX,ext_pixelsY))) #   MATRIX 
                                                            #   of pixels color
#as above!
    int_pixelGrid=ext_pixelGrid #calculates list of all pixels new coordinates
    int_circle=poi(int_figPosList[cir-1][0]).buffer(cirRad) 
    #insert row for the new trial retina's position
    int_retiPixALL=np.vstack((int_retiPixALL,np.matrix(np.zeros(int_pixelsX*int_pixelsY))))
    int_retiPixVec=int_retiPixALL[numTri,:]

    pixelLoop(int_pixelGrid,int_objTri.xy,int_objSqu.xy,int_circle,int_retiPixVec)       
    int_retiPixMat=np.flipud(np.reshape(int_retiPixVec, (int_pixelsX,int_pixelsY))) #   MATRIX 
                                                            #   of pixels color    
    
#!!!!!!!#_________________________________________@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    #CHECK if hand is inside the desk   
    #NO
    if (poi(objHand.xy).within(poly(objDesk.xy))) != True:
        ext_figPosList[hand][numTri]=ext_figPosList[hand][numTri-1]
        continue
    
    #YES --------->          #CHECK if hand is inside the figure
    else:
        if ((poi(objHand.xy).within(poly(ext_objTri.xy))) == True):
            figMov=tri   
        elif ((poi(objHand.xy).within(poly(ext_objSqu.xy))) == True):
            figMov=squ    
        elif ((poi(objHand.xy).within(ext_circle)) == True):
            figMov=cir   
        else:
            continue
    
    #if INSIDE the desk and the figure ------>
    action.append(ext_figPosList[hand][numTri])
    
    #search new position:   
    while True:
        numMov+=1
        ext_figPosList[hand].append(objHand.randPos())
        handCoor=ext_figPosList[hand][numMov]               
        objHand.xy=handCoor
        
        #CHECK if hand is out of the desk
        #NO
        if (poi(objHand.xy).within(poly(objDesk.xy))) != True:
            ext_figPosList[hand][numMov]=ext_figPosList[hand][numMov-1]
            ext_figPosList[tri].append(ext_figPosList[tri][numMov-1]) 
            ext_figPosList[squ].append(ext_figPosList[squ][numMov-1])
            ext_figPosList[cir].append(ext_figPosList[cir][numMov-1])   
            
            ext_retiCoorList.append(ext_retiCoor)    
            ext_foveaList.append(ext_foveaCoor)
            
            int_retiCoorList.append(ext_retiCoor)
            ext_foveaList.append(ext_foveaCoor)

            
        #YES -------->
        else:                             
            # -------> MOVE!
            figVectMov=np.zeros([4,2]) #movement vector

            move=ext_figPosList[hand][numMov]-ext_figPosList[hand][numMov-1]
            figVectMov[figMov]+=move #updated movement vectore
            
            for figure in range (1,4):
                num = ext_figPosList[figure][-1]+figVectMov[figure] #-1 gets the last element of the list
                ext_figPosList[figure].append(num)

            ext_objTri.xy=ext_figPosList[tri][numMov] 
            ext_objSqu.xy=ext_figPosList[squ][numMov]
            ext_objCir.xy=ext_figPosList[cir][numMov]
         
            ext_retiCoorList.append(ext_retiCoor)   
            ext_foveaList.append(ext_foveaCoor)
            
            int_retiCoorList.append(ext_retiCoor)
            ext_foveaList.append(ext_foveaCoor)


            #CHECK if figure is in/out of the desk
            ext_circle=poi(ext_objCir.xy).buffer(cirRad)
            if (poly(ext_objTri.xy).within(poly(objDesk.xy))) != True\
              or (poly(ext_objSqu.xy).within(poly(objDesk.xy)))!= True\
              or ((ext_circle).within(poly(objDesk.xy))) != True:

                ext_figPosList[hand][-1]=ext_figPosList[hand][numMov-1]
                objHand.xy=ext_figPosList[hand][numMov-1]

                ext_figPosList[tri][-1]=ext_figPosList[tri][numMov-1]
                ext_objTri.xy=ext_figPosList[tri][numMov-1]
                
                ext_figPosList[squ][-1]=ext_figPosList[squ][numMov-1]
                ext_objSqu.xy=ext_figPosList[squ][numMov-1]
                
                ext_figPosList[cir][-1]=ext_figPosList[cir][numMov-1]
                ext_objCir.xy=ext_figPosList[cir][numMov-1]
                continue
            
            else:
                #SOON ---> insert new saccade?!
                #for now saccade num = hand movements num EXCEPT for the last
                #movement of "release" of the object
                print("\nMOVE, at step: ", numTri)
                action.append(objHand.xy)
                
                #ADD CHECK WITH INTERNAL RETINA(GOAL)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                break
                             
    
#ANIMATE
if graphic == "on":
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_________A_N_I_M_A_T_I_O_N_______%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    plot_extSet=1
    plot_extReti=2
    plot_intSet=3
    plot_intReti=4
    
            #_______right SEQUENCE of figures on the plot
    plotIndex=[0,1,2,3]
    plotIndex[figMov],plotIndex[-1]=plotIndex[-1],plotIndex[figMov] #switch
    plotIndex.append(plotIndex.pop(plotIndex.index(0))) #hand to last position
    
    plotObj=[objHand,ext_objTri,ext_objSqu,ext_objCir]
    plotObj[figMov],plotObj[-1]=plotObj[-1],plotObj[figMov] #switch
    plotObj.append(plotObj.pop(plotObj.index(objHand))) #hand to last position
    
    def plotOnScreen(x,y,w,h): #define position of plot-windows on the screen
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(x,y,w,h) 
    def ext_initDesk():
        objDesk.plotDef()
        objDesk.plotAdd(plot_extSet,"External Environment")  
        ext_objTri.plotDef()
        ext_objTri.plotAdd(plot_extSet,"External Environment")
        ext_objSqu.plotDef()
        ext_objSqu.plotAdd(plot_extSet,"External Environment")
        ext_objCir.plotDef()
        ext_objCir.plotAdd(plot_extSet,"External Environment")
        objHand.plotDef()
        objHand.plotAdd(plot_extSet,"External Environment")
        ext_objReti.plotDef()
        ext_objReti.plotAdd(plot_extSet,"External Environment")
        ext_objFovea.plotDef()
        ext_objFovea.plotAdd(plot_extSet,"External Environment")
    
    def int_initDesk():

        objDesk.plotDef()
        objDesk.plotAdd(plot_intSet,"Goal")  
        int_objTri.plotDef()
        int_objTri.plotAdd(plot_intSet,"Goal")
        int_objSqu.plotDef()
        int_objSqu.plotAdd(plot_intSet,"Goal")
        int_objCir.plotDef()
        int_objCir.plotAdd(plot_intSet,"Goal")
        
        int_objReti.plotDef()
        int_objReti.plotAdd(plot_intSet,"Goal")
        
        int_objFovea.plotDef()
        int_objFovea.plotAdd(plot_intSet,"Goal")
        

        #_______EXT_SETTING    
    def animSetExt(position):
        ext_objFovea.animateFig(ext_foveaList,position,1,"External Environment")
        ext_objReti.animateFig(ext_retiCoorList,position,1,"External Environment")
        
        plotObj[0].animateFig(ext_figPosList[plotIndex[0]],position,plot_extSet,"External Environment")  
        plotObj[1].animateFig(ext_figPosList[plotIndex[1]],position,plot_extSet,"External Environment")  
        plotObj[2].animateFig(ext_figPosList[plotIndex[2]],position,plot_extSet,"External Environment")  
        plotObj[3].animateFig(ext_figPosList[plotIndex[3]],position,plot_extSet,"External Environment")

    fig1=plt.figure(1) 
    anim = animation.FuncAnimation(fig1, animSetExt, frames=len(ext_figPosList[0]),
                                   init_func=ext_initDesk,
                                   repeat=False, interval=800)
    plotOnScreen(100,10,500,500) 
      
      #______EXT_RETINA
    fig2=plt.figure(2)
    ax2 = fig2.add_subplot(111,aspect='equal')  
    ax2.set_title("External Retina")
    
    def animRetiExt(position2):
        fig2=plt.figure(2)
    
        retiPixPLOT=np.flipud(np.reshape(ext_retiPixALL[position2,:], (ext_pixelsX,ext_pixelsY)))
    
        fig2, plt.imshow(retiPixPLOT, interpolation='nearest')
        plt.grid(True)  
    
    anim2 = animation.FuncAnimation(fig2, animRetiExt, frames=len(ext_retiPixALL), #ADD INIT DESK
                                   repeat=False, interval=800)
    plotOnScreen(600,10,500,500) 
    
        #_____INT_SETTING
    def animSetInt(position3):
        int_objFovea.animateFig(int_foveaList,position3,plot_intSet,"Goal")
        int_objReti.animateFig(int_retiCoorList,position3,plot_intSet,"Goal")
        
    fig3=plt.figure(3)
    anim3 = animation.FuncAnimation(fig3, animSetInt, frames=len(ext_figPosList[0]),
                                   init_func=int_initDesk,
                                   repeat=False, interval=800)

    plotOnScreen(100,600,500,500)                              
                               
        #____INT_RETINA
                               
    fig4=plt.figure(4)
    ax4 = fig4.add_subplot(111,aspect='equal')  
    ax4.set_title("Internal Retina")
    
    def animRetiInt(position4):
        fig4=plt.figure(4)
    
        retiPixPLOT=np.flipud(np.reshape(int_retiPixALL[position4,:], (int_pixelsX,int_pixelsY)))
    
        fig4, plt.imshow(retiPixPLOT, interpolation='nearest')
        plt.grid(True)  
    
    anim4 = animation.FuncAnimation(fig4, animRetiInt, frames=len(int_retiPixALL), #ADD INIT DESK
                                   repeat=False, interval=800)
    plotOnScreen(600,600,500,500) 
                               
#NOTE THAT WHEN THE HAND "GRASPS" THE FIGURE THE RETINA IS STILL ON THE PREVIOUS POSITION
# OF THE HAND ACTION 0 (WHEN THE HAND "GRASPS" THE FIGURE), SO IT WON'T MOVE TO THE
#NEW HAND ACTION 1 (WHEN THE HAND "RELEASE" THE FIGURE)

"""TO DO:
1. OK - Change definition of ext_retiPixMat so that it accepts also resolutions with
    different axes (es. not 11x11 pixels but 11x15, 25x11 ecc)
2. OK - Use "shapely" instead of checkPath/checkPoint?
3. Calculate new random fovea's position when hand releases the object?
    3b. Does the eye have to move together with the hand (hand xy = fovea xy)..?
4. Insert new function for extract the pixel grid?
5. OPTIMIZATION of classes/functions/code!
    5b. Vectorization of pixel grid to avoid the use of the for loop?! (the program
    becomes too slow with more resolution)
6. Suggestions? :P
"""
