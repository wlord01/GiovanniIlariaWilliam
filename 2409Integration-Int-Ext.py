# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:29:15 2017

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

class Polygon(object):
    
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
                 
    def plotAdd(self): #add plot to figure
        self.xLim=(-5,15)      
        self.yLim=(-5,15)
        self.fig=plt.figure(1) 
        self.ax1 = self.fig.add_subplot(111,aspect='equal')       
        self.ax1.set_xlim((self.xLim))
        self.ax1.set_ylim((self.yLim))
        
        self.ax1.add_patch(self.figure)
        
    def animateFig(self,coorList,position): #plot animation
        self.xy=coorList[position]
        self.figure.remove()
        self.plotDef()
        self.plotAdd()
        
        
class Circle(Polygon):
    
    def __init__(self,xy,radius):
        self.xy=xy
        self.radius=radius
        self.color="none"

    def plotDef(self):
        self.figure=ptc.Circle((self.xy),self.radius, 
                            facecolor = self.color)

class Hand(Circle):
    def __init__(self,xy,radius):
        Circle.__init__(self,xy,radius)
        
class Retina(Polygon):
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
        
    def plotAdd(self):
        #modified, to plot the grid of retina on the setting
        self.xLim=(-5,15)      
        self.yLim=(-5,15)
        self.fig=plt.figure(1) 
        self.ax1 = self.fig.add_subplot(111,aspect='equal')       
        self.ax1.set_xlim((self.xLim))
        self.ax1.set_ylim((self.yLim))
        
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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%____DATA___INITIALIZATION
#COLORS for eye plot
green=1
red=2
blue=3
others=0

#FIGURES - INDICES/LISTS
hand=0
tri=1
squ=2
cir=3

figMov=0 #figure that moves
figVectMov=np.zeros([4,2]) #vector of all figures movement
# or -----> figPosMat=np.matrix([handCoor,ext_triCoor,ext_squCoor,ext_cirCoor])

#DESK
deskCoor=np.array([[1,1],[9,1],[9,9],[1,9]],dtype=float)

objDesk=Polygon(deskCoor)
objDesk.color="none"

#___________________________________________EXTERNAL_ENVIRONMENT___%%%%%%%%%%%%
#TRIANGLE
ext_triCoor=np.array([[1.5,6.5],[3,6.5],[2.25,8]])

ext_objTri=Polygon(ext_triCoor)
ext_objTri.color="g"

#SQUARE
ext_squCoor=np.array([[4,6.5],[5.5,6.5],[5.5,8],[4,8]])

ext_objSqu=Polygon(ext_squCoor)
ext_objSqu.color="r"

#CIRCLE
ext_cirCoor=np.array([7.2,7.2])
cirRad=0.8

ext_objCir=Circle(ext_cirCoor,cirRad)
ext_objCir.color="b"

#HAND
handCoor=np.array([5,1.2])
handRad=0.1
action=[]
#actMat=np.matrix(np.zeros([2,2])) #action matrix (x0,y0;x1,y1)
objHand=Hand(handCoor,handRad)
objHand.color="y"

ext_figPosList=[[handCoor],[ext_triCoor],[ext_squCoor],[ext_cirCoor]] #list of all figures positions

#___________________________________________EXTERNAL_RETINA___%%%%%%%%%%%%%%%%%
#RETINA
pixelsX=30
pixelsY=30

retiCoor=deskCoor
retiCoorList=[]
retiCoorList.append(retiCoor)

#matrix of x(trial) * y(vector of pixels)
retiPixALL=np.matrix(np.zeros(pixelsX*pixelsY))
#matrix of x*y pixels of actual trial (es. matrix 11x11 of trial n)
retiPixMat=np.matrix(np.zeros([pixelsX,pixelsY]))
#vector of x*y pixels of actual trial (es. vector 121 of trial n)
retiPixVec=retiPixALL[0,:]

objReti=Retina(retiCoor,pixelsX,pixelsY)
objReti.linestyle=":"
objReti.edge="k"

#_______________________extract the PIXELS GRID

pixelGrid=objReti.pixelCalc() #calculates list of all pixels coordinates

#this is the definition of the circle to allow the use of "shapely" library
#(to check if pixel polygon intersecates the circle polygon)
#NO need to define it for the other polygons (triangle, square)
#-----> see below 
circle=poi(ext_figPosList[cir][0]).buffer(cirRad) 
def pixelLoop():
    cell=-1 #pixel
    for pix in (pixelGrid): #for every pixel of the retina's grid --->
        cell+=1
        
        #___check if PIXEL is on a FIGURE, then: 
                                          #   add the value RGB in retinaVector
                                          #   depending on which is the figure 
        
        #----->below..
        if (poly(ext_objTri.xy).intersects(poly(pix)))==True:
            retiPixVec[0,cell]=green
        elif (poly(ext_objSqu.xy).intersects(poly(pix)))==True:
            retiPixVec[0,cell]=red
        elif ((circle).intersects(poly(pix))) == True:
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

pixelLoop()               
retiPixMat=np.flipud(np.reshape(retiPixVec, (pixelsX,pixelsY))) #   MATRIX 
                                                            #   of pixels color
                        #np.flipud is necessary to not have the matrix reversed ;)

#_______________________________FOVEA

retiFovea=objReti.calcFovea() #calculates fovea xy
objFovea=Circle(retiFovea, 0.05)

foveaList=[retiFovea] #list of all fovea's random positions


'''
#___________________________________INTERNAL_ENVIRONMENT(GOAL)___%%%%%%%%%%%%%%
#TRIANGLE
triCoorInt=np.array([[1.5,6.5],[3,6.5],[2.25,8]])

objTriInt=Polygon(ext_triCoor)
objTriInt.color="g"

#SQUARE
squCoorInt=np.array([[6,4.5],[7.5,4.5],[7.5,6],[6,6]])

objSquInt=Polygon(ext_squCoor)
objSquInt.color="r"

#CIRCLE
cirCoorInt=np.array([7.2,7.2])

objCirInt=Circle(ext_cirCoor,cirRad)
objCirInt.color="b"

figPosListInt=[[ext_triCoor],[ext_squCoor],[ext_cirCoor]] #list of all figures positions

#___________________________________________INTERNAL_RETINA___%%%%%%%%%%%%%%%%%
#RETINA
pixelsX=30
pixelsY=30

retiCoor=deskCoor
retiCoorList=[]
retiCoorList.append(retiCoor)

#matrix of x(trial) * y(vector of pixels)
retiPixALL=np.matrix(np.zeros(pixelsX*pixelsY))
#matrix of x*y pixels of actual trial (es. matrix 11x11 of trial n)
retiPixMat=np.matrix(np.zeros([pixelsX,pixelsY]))
#vector of x*y pixels of actual trial (es. vector 121 of trial n)
retiPixVec=retiPixALL[0,:]

objReti=Retina(retiCoor,pixelsX,pixelsY)
objReti.linestyle=":"
objReti.edge="k"

#_______________________extract the PIXELS GRID

pixelGrid=objReti.pixelCalc() #calculates list of all pixels coordinates

#this is the definition of the circle to allow the use of "shapely" library
#(to check if pixel polygon intersecates the circle polygon)
#NO need to define it for the other polygons (triangle, square)
#-----> see below 
circle=poi(ext_figPosList[cir][0]).buffer(cirRad) 

cell=-1 #pixel

for pix in (pixelGrid): #for every pixel of the retina's grid --->
    cell+=1
    
    #___check if PIXEL is on a FIGURE, then: 
                                      #   add the value RGB in retinaVector
                                      #   depending on which is the figure 
    
    #----->below..
    if (poly(ext_objTri.xy).intersects(poly(pix)))==True:
        retiPixVec[0,cell]=green
    elif (poly(ext_objSqu.xy).intersects(poly(pix)))==True:
        retiPixVec[0,cell]=red
    elif ((circle).intersects(poly(pix))) == True:
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
           
retiPixMat=np.flipud(np.reshape(retiPixVec, (pixelsX,pixelsY))) #   MATRIX 
                                                        #   of pixels color
                    #np.flipud is necessary to not have the matrix reversed ;)

#_______________________________FOVEA

retiFovea=objReti.calcFovea() #calculates fovea xy
objFovea=Circle(retiFovea, 0.05)

foveaList=[retiFovea] #list of all fovea's random positions
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%___START
graphic="off"
#TRIALS
numTri=0
numMov=0
maxTri=50
while numTri < maxTri and ((poi(objHand.xy).within(poly(ext_objTri.xy))) != True)\
                      and ((poi(objHand.xy).within(poly(ext_objSqu.xy))) != True)\
                      and ((poi(objHand.xy).within(circle)) != True):
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
    foveaList.append(objReti.randPos()) #new random fovea position
    foveaCoor=foveaList[numTri]
    
         #___check if FOVEA is inside the desk
    while True:
        #if NO ---> calculate new fovea position..
        if (poi(foveaCoor).within(poly(objDesk.xy))) != True: 
            foveaList[numTri]=objReti.randPos()
            foveaCoor=foveaList[numTri]
        #..until YES! ---> fovea is inside the desk
        else:
            break   
        
        #___MOVE RETINA
    moveFove=foveaList[numTri] - foveaList[numTri-1] #calculate fovea movement
    retiCoor=retiCoor+moveFove.T #calculate new position of retina vertices
    retiCoorList.append(retiCoor)    
    objReti.xy=retiCoor
    
#!!!!!!!#___extract the PIXELS GRID_______________@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #as above!
    pixelGrid=objReti.pixelCalc() #calculates list of all pixels new coordinates
     
    circle=poi(ext_figPosList[cir][numTri]).buffer(cirRad) 

    #insert row for the new trial retina's position
    retiPixALL=np.vstack((retiPixALL,np.matrix(np.zeros(pixelsX*pixelsY))))
    retiPixVec=retiPixALL[numTri,:]

    pixelLoop() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
    retiPixMat=np.flipud(np.reshape(retiPixVec, (pixelsX,pixelsY))) #   MATRIX 
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
        elif ((poi(objHand.xy).within(circle)) == True):
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
            
            retiCoorList.append(retiCoor)    
            foveaList.append(foveaCoor)
            
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
            
            
            retiCoorList.append(retiCoor)   
            foveaList.append(foveaCoor)


            #CHECK if figure is in/out of the desk
            circle=poi(ext_objCir.xy).buffer(cirRad)
            if (poly(ext_objTri.xy).within(poly(objDesk.xy))) != True\
              or (poly(ext_objSqu.xy).within(poly(objDesk.xy)))!= True\
              or ((circle).within(poly(objDesk.xy))) != True:

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
                print "\nMOVE, at step: ", numTri
                action.append(objHand.xy)
                
                #ADD CHECK WITH INTERNAL RETINA(GOAL)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                break
            
   
#ANIMATE
if graphic == "on":
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%___ANIMATION___INITIALIZATION
    def initDesk():
        objDesk.plotDef()
        objDesk.plotAdd()  
        ext_objTri.plotDef()
        ext_objTri.plotAdd()
        ext_objSqu.plotDef()
        ext_objSqu.plotAdd()
        ext_objCir.plotDef()
        ext_objCir.plotAdd()
        objHand.plotDef()
        objHand.plotAdd()
        objReti.plotDef()
        objReti.plotAdd()
        objFovea.plotDef()
        objFovea.plotAdd()
        #_______right SEQUENCE of figures on the plot
    plotIndex=[0,1,2,3]
    plotIndex[figMov],plotIndex[-1]=plotIndex[-1],plotIndex[figMov] #switch
    plotIndex.append(plotIndex.pop(plotIndex.index(0))) #hand to last position
    
    plotObj=[objHand,ext_objTri,ext_objSqu,ext_objCir]
    plotObj[figMov],plotObj[-1]=plotObj[-1],plotObj[figMov] #switch
    plotObj.append(plotObj.pop(plotObj.index(objHand))) #hand to last position
    
        #_______SETTING
    
    def animSetExt(position):
        objFovea.animateFig(foveaList,position)
        objReti.animateFig(retiCoorList,position)
        
        plotObj[0].animateFig(ext_figPosList[plotIndex[0]],position)  
        plotObj[1].animateFig(ext_figPosList[plotIndex[1]],position)  
        plotObj[2].animateFig(ext_figPosList[plotIndex[2]],position)  
        plotObj[3].animateFig(ext_figPosList[plotIndex[3]],position)
        
    fig=plt.figure(1) 
    anim = animation.FuncAnimation(fig, animSetExt, frames=len(ext_figPosList[0]),
                                   init_func=initDesk,
                                   repeat=False, interval=400)
                                   
        #______RETINA
    fig2=plt.figure(2)
    ax2 = fig2.add_subplot(111,aspect='equal')  
    
    def animRetiExt(position2):
        fig2=plt.figure(2)
    
        retiPixPLOT=np.flipud(np.reshape(retiPixALL[position2,:], (pixelsX,pixelsY)))
    
        fig2, plt.imshow(retiPixPLOT, interpolation='nearest')
        plt.grid(True)  
    
    anim2 = animation.FuncAnimation(fig2, animRetiExt, frames=len(retiPixALL), #ADD INIT DESK
                                   repeat=False, interval=400)
                                   

                               
#NOTE THAT WHEN THE HAND "GRASPS" THE FIGURE THE RETINA IS STILL ON THE PREVIOUS POSITION
# OF THE HAND ACTION 0 (WHEN THE HAND "GRASPS" THE FIGURE), SO IT WON'T MOVE TO THE
#NEW HAND ACTION 1 (WHEN THE HAND "RELEASE" THE FIGURE)

"""TO DO:
1. OK - Change definition of retiPixMat so that it accepts also resolutions with
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