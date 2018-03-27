# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:27:14 2018

@author: plin@augusta.edu
"""

import numpy as np
import cv2
from numpy.linalg import norm as nrm
from LineDraw import *
import matplotlib.pyplot as plt

btn_down = False

def get_points(img):
    params = {}
    params['img'] = img.copy()
    params['lines'] = []
        
    # set the callback function for any mouse event
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mouse_handler, params)
    cv2.waitKey(0)
        
    # convert array to np.array with a shape of [n,2,2]
    Endpoints = np.uint16(params['lines'])
        
    return Endpoints, params['img']

def VOIandSeeds(points, img, thk = 3, crop = 0.15, showImg = True):
    """ calculate the midpoint of the line segment and define it as the centre of a circle
        define the diameter of the circle (or sphere) and then select a VOI that is a bounding box of 
        the circle (sphere) with a diameter 35% longer than the line segment.
    """
    # this version works for 2D images only
    fg_seeds, bg_seeds = [], []
    if len(points) != 1:
        print('Only a single line drawn is allowed!')
        pass
       
    ctr_x, ctr_y = np.mean(points[0,:,:], axis=0).astype(int)  # x, y coordinates of the centre
    box_length = int(np.linalg.norm(points[0,0,:].astype(int) - points[0,1,:].astype(int))*1.35)  # 1.35x diameter
    x_upLeft = ctr_x - int(box_length/2)
    y_upLeft = ctr_y - int(box_length/2)
    
    x_dwnRight = ctr_x + int(box_length/2)
    y_dwnRight = ctr_y + int(box_length/2)
    
    if x_upLeft < 0:
        x_upLeft = 0
    if y_upLeft < 0:
        y_upLeft = 0
    if x_dwnRight > (img.shape[1]-1):
        x_dwnRight = img.shape[1]-1
    if y_dwnRight > (img.shape[0]-1):
        y_dwnRight = img.shape[0]-1    
    
    # background seeds
    bg_seeds = np.concatenate((np.flip(np.insert(np.expand_dims(np.array(np.arange(y_upLeft, y_dwnRight+1)), axis=1), 1, x_upLeft, axis=1),1),\
                          np.flip(np.insert(np.expand_dims(np.array(np.arange(y_upLeft, y_dwnRight+1)), axis=1), 1, x_dwnRight, axis=1),1),\
                          np.insert(np.expand_dims(np.array(np.arange(x_upLeft, x_dwnRight+1)), axis=1), 1, y_upLeft, axis=1),\
                          np.insert(np.expand_dims(np.array(np.arange(x_upLeft, x_dwnRight+1)), axis=1), 1, y_dwnRight, axis=1)), axis=0)

    #forgroud seeds
    fg_seeds = line_seg_points(points[0,:].astype(int), thk, crop)

    # draw a green rectangle to visualize the bounding rect
    
    cv2.rectangle(img, (x_upLeft, y_upLeft), (x_dwnRight, y_dwnRight), (0, 255, 0), 1)
    cv2.polylines(img, [bg_seeds],True,(255,0,0),lineType= 8)
#    cv2.polylines(img, [fg_seeds], True, (0,0,255))
    if showImg:
        plt.plot(fg_seeds[:,0], fg_seeds[:,1],'r', linewidth = 1.0)
#    cv2.imshow('Image', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if showImg:
        plt.imshow(img)
        cv2.waitKey(0)
    
    seeds = (fg_seeds, bg_seeds)
    return points, seeds, img

def mouse_handler(event, x, y, flags, params):
    global btn_down
        
    if event == cv2.EVENT_LBUTTONUP and btn_down:
        # if the mouse button released, draw the line
        btn_down = False
        params['lines'][0].append((x,y)) # appending the second point
        cv2.circle(params['img'], (x, y), 1, (0,0,255), -1)
        cv2.line(params['img'], params['lines'][0][0], params['lines'][0][1], (255,255,255), 1)
        cv2.imshow('Image', params['img'])
            
    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        # synchronizing line visualization
        IMG = params['img'].copy()
        cv2.line(IMG, params['lines'][0][0], (x, y), (0,255,0), 1)
        cv2.imshow('Image', IMG)
        
    elif event == cv2.EVENT_LBUTTONDOWN and len(params['lines']) < 1:
        btn_down = True
        params['lines'].insert(0, [(x, y)]) # prepend the point
        cv2.circle(params['img'], (x, y), 1, (0,0,255), -1)
        cv2.imshow('Image', params['img'])
        
        
def line_seg_points(points, thk, crop):
    #find all points on the line segment defined by points (start_point and end_point)
    #the line segment is cropped  by 15% from each end
    w, h = np.diff(points, axis=0)[0]
                        
    if w != 0:
        slope = h / w   
    else:
        slope = np.inf

    start_point = points[0] + (np.array([w, h])*crop).astype(int)
    end_point = points[1] - (np.array([w, h])*crop).astype(int)
#  if thk = 1, pts is the solution (may be removed later)
#    pts = connect(np.array([start_point, end_point]))
    
    if slope == 0:
        # h = 0; therefore, slope_norm = inf (w_norm = 0)
        slope_norm = np.inf
    elif slope != 0:
        slope_norm = -w / h
    
    # generate two matrices containg the start points and end points after cropping
    start_pts = start_point + np.array([np.round(np.cos(np.arctan(slope_norm))*(np.arange(thk) - int(thk/2))),\
            np.round(np.sin(np.arctan(slope_norm))*(np.arange(thk) - int(thk/2)))]).T
    end_pts = end_point + np.array([np.round(np.cos(np.arctan(slope_norm))*(np.arange(thk) - int(thk/2))),\
            np.round(np.sin(np.arctan(slope_norm))*(np.arange(thk) - int(thk/2)))]).T   
        
    ## apply start_pts and end_pts to connect(ends) or connect2(ends) function
    output_pts = Murphy_line_vectorize(start_pts, end_pts, thk)
    return output_pts
