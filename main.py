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
from shapely.geometry import Polygon as poly #for check pixel/figure intersection
from shapely.geometry import Point as poi

import ini_obj as ini

plt.ion()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_____________S_T_A_R_T___________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%_________________________________%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#FIGURES - INDICES/LISTS
hand,tri,squ,cir=0,1,2,3
graphic="on"
#TRIALS
numTri=0
numMov=0
maxTri=50
while numTri < maxTri and ((poi(ini.objHand.xy).within(poly(ini.ext_objTri.xy))) != True)\
                      and ((poi(ini.objHand.xy).within(poly(ini.ext_objSqu.xy))) != True)\
                      and ((poi(ini.objHand.xy).within(ini.ext_circle)) != True):
                      #MODIFY WITH CHECKING IF EXTERNAL RETINA != INTERNAL RETINA!!!!!!!!!!!!!!!!!!
                          
    numTri+=1
    numMov+=1

    #New random hand position    
    ini.ext_figPosList[hand].append(ini.objHand.randPos())
    handCoor=ini.ext_figPosList[hand][numTri]
    ini.objHand.xy=handCoor

    #New position set
    ini.ext_figPosList[tri].append(ini.ext_triCoor)
    ini.ext_figPosList[squ].append(ini.ext_squCoor)
    ini.ext_figPosList[cir].append(ini.ext_cirCoor)
     
    
    #New random RETINA position
    ini.ext_foveaList.append(ini.ext_objReti.randPos()) #new random fovea position
    ext_foveaCoor=ini.ext_foveaList[numTri]
    
    ini.int_foveaList.append(ini.ext_foveaList[-1])
    int_foveaCoor=ini.int_foveaList[numTri]
    
    
         #___check if FOVEA is inside the desk
    while True:
        #if NO ---> calculate new fovea position..
        if (poi(ext_foveaCoor).within(poly(ini.objDesk.xy))) != True: 
            ini.ext_foveaList[numTri]=ini.ext_objReti.randPos()
            ext_foveaCoor=ini.ext_foveaList[numTri]
            
            ini.int_foveaList[numTri]=ini.ext_foveaList[numTri]
            int_foveaCoor=ini.int_foveaList[numTri]
        #..until YES! ---> fovea is inside the desk
        else:
            break   
        
        #___MOVE RETINA
    moveFove=ini.ext_foveaList[numTri] - ini.ext_foveaList[numTri-1] #calculate fovea movement
    ini.ext_retiCoor=ini.ext_retiCoor+moveFove.T #calculate new position of retina vertices
    ini.ext_retiCoorList.append(ini.ext_retiCoor)    
    ini.ext_objReti.xy=ini.ext_retiCoor
    
    ini.int_retiCoor=ini.int_retiCoor+moveFove.T
    ini.int_retiCoorList.append(ini.int_retiCoor)
    ini.int_objReti.xy=ini.int_retiCoor
    
#!!!!!!!#___extract the PIXELS GRID_______________@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #as above!
    ext_pixelGrid=ini.ext_objReti.pixelCalc() #calculates list of all pixels new coordinates
    ini.ext_circle=poi(ini.ext_figPosList[cir][numTri]).buffer(ini.cirRad) 
    #insert row for the new trial retina's position
    ini.ext_retiPixALL=np.vstack((ini.ext_retiPixALL,np.matrix(np.zeros(ini.ext_pixelsX*ini.ext_pixelsY))))
    ext_retiPixVec=ini.ext_retiPixALL[numTri,:]

    ini.Retina.pixelLoop(ext_pixelGrid,ini.ext_objTri.xy,ini.ext_objSqu.xy,ini.ext_circle,ext_retiPixVec)       
    ext_retiPixMat=np.flipud(np.reshape(ext_retiPixVec, (ini.ext_pixelsX,ini.ext_pixelsY))) #   MATRIX 
                                                            #   of pixels color
#as above!
    int_pixelGrid=ext_pixelGrid #calculates list of all pixels new coordinates
    int_circle=poi(ini.int_figPosList[cir-1][0]).buffer(ini.cirRad) 
    #insert row for the new trial retina's position
    ini.int_retiPixALL=np.vstack((ini.int_retiPixALL,np.matrix(np.zeros(ini.int_pixelsX*ini.int_pixelsY))))
    int_retiPixVec=ini.int_retiPixALL[numTri,:]

    ini.Retina.pixelLoop(int_pixelGrid,ini.int_objTri.xy,ini.int_objSqu.xy,int_circle,int_retiPixVec)       
    int_retiPixMat=np.flipud(np.reshape(int_retiPixVec, (ini.int_pixelsX,ini.int_pixelsY))) #   MATRIX 
                                                            #   of pixels color    
    
#!!!!!!!#_________________________________________@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    #CHECK if hand is inside the desk   
    #NO
    if (poi(ini.objHand.xy).within(poly(ini.objDesk.xy))) != True:
        ini.ext_figPosList[hand][numTri]=ini.ext_figPosList[hand][numTri-1]
        continue
    
    #YES --------->          #CHECK if hand is inside the figure
    else:
        if ((poi(ini.objHand.xy).within(poly(ini.ext_objTri.xy))) == True):
            ini.figMov=tri   
        elif ((poi(ini.objHand.xy).within(poly(ini.ext_objSqu.xy))) == True):
            ini.figMov=squ    
        elif ((poi(ini.objHand.xy).within(ini.ext_circle)) == True):
            ini.figMov=cir   
        else:
            continue
    
    #if INSIDE the desk and the figure ------>
    ini.action.append(ini.ext_figPosList[hand][numTri])
    
    #search new position:   
    while True:
        numMov+=1
        ini.ext_figPosList[hand].append(ini.objHand.randPos())
        handCoor=ini.ext_figPosList[hand][numMov]               
        ini.objHand.xy=handCoor
        
        #CHECK if hand is out of the desk
        #NO
        if (poi(ini.objHand.xy).within(poly(ini.objDesk.xy))) != True:
            ini.ext_figPosList[hand][numMov]=ini.ext_figPosList[hand][numMov-1]
            ini.ext_figPosList[tri].append(ini.ext_figPosList[tri][numMov-1]) 
            ini.ext_figPosList[squ].append(ini.ext_figPosList[squ][numMov-1])
            ini.ext_figPosList[cir].append(ini.ext_figPosList[cir][numMov-1])   
            
            ini.ext_retiCoorList.append(ini.ext_retiCoor)    
            ini.ext_foveaList.append(ext_foveaCoor)
            
            ini.int_retiCoorList.append(ini.ext_retiCoor)
            ini.ext_foveaList.append(ext_foveaCoor)

            
        #YES -------->
        else:                             
            # -------> MOVE!
            ini.figVectMov=np.zeros([4,2]) #movement vector

            move=ini.ext_figPosList[hand][numMov]-ini.ext_figPosList[hand][numMov-1]
            ini.figVectMov[ini.figMov]+=move #updated movement vectore
            
            for figure in range (1,4):
                num = ini.ext_figPosList[figure][-1]+ini.figVectMov[figure] #-1 gets the last element of the list
                ini.ext_figPosList[figure].append(num)

            ini.ext_objTri.xy=ini.ext_figPosList[tri][numMov] 
            ini.ext_objSqu.xy=ini.ext_figPosList[squ][numMov]
            ini.ext_objCir.xy=ini.ext_figPosList[cir][numMov]
         
            ini.ext_retiCoorList.append(ini.ext_retiCoor)   
            ini.ext_foveaList.append(ext_foveaCoor)
            
            ini.int_retiCoorList.append(ini.ext_retiCoor)
            ini.ext_foveaList.append(ext_foveaCoor)


            #CHECK if figure is in/out of the desk
            ini.ext_circle=poi(ini.ext_objCir.xy).buffer(ini.cirRad)
            if (poly(ini.ext_objTri.xy).within(poly(ini.objDesk.xy))) != True\
              or (poly(ini.ext_objSqu.xy).within(poly(ini.objDesk.xy)))!= True\
              or ((ini.ext_circle).within(poly(ini.objDesk.xy))) != True:

                ini.ext_figPosList[hand][-1]=ini.ext_figPosList[hand][numMov-1]
                ini.objHand.xy=ini.ext_figPosList[hand][numMov-1]

                ini.ext_figPosList[tri][-1]=ini.ext_figPosList[tri][numMov-1]
                ini.ext_objTri.xy=ini.ext_figPosList[tri][numMov-1]
                
                ini.ext_figPosList[squ][-1]=ini.ext_figPosList[squ][numMov-1]
                ini.ext_objSqu.xy=ini.ext_figPosList[squ][numMov-1]
                
                ini.ext_figPosList[cir][-1]=ini.ext_figPosList[cir][numMov-1]
                ini.ext_objCir.xy=ini.ext_figPosList[cir][numMov-1]
                continue
            
            else:
                #SOON ---> insert new saccade?!
                #for now saccade num = hand movements num EXCEPT for the last
                #movement of "release" of the object
                print("\nMOVE, at step: ", numTri)
                ini.action.append(ini.objHand.xy)
                
                #ADD CHECK WITH INTERNAL RETINA(GOAL)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                break
                         
    
#ANIMATE
if graphic == "on": 
    import anim_plot

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
