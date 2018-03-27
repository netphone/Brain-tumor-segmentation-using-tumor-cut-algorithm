# -*- coding: utf-8 -*-
"""
Created on Sat Mar  20 11:20:00 2018

@author: Power-R7
"""

import numpy as np
from scipy import ndimage


def sdf(prob_img, zero_level = 0.5):
    """ The signed distance function (sdf) is a level set function that gives the shortest distance
    to the nearest point on the interface.
       
    Typically, there is a level set function f(x,y) that gives the interface as the set of points for which
    f(x,y)=0. This is called the ‘zero level set’, and in this case, the signed distance function phi(x,y)
    can be initialized using phi(x,y)=f(x,y)

    Given the interface presented by the zero level set function, we need every point (pixel) of the
    signed distance function outside of the interface to have a value equal to have a value equal to
    its distance from the interface.
    """   

    # construct_sdf takes a level set function (lsf) as input and outputs the corresponding signed distance function (sdf)
    # sign function presenting the value of each pixel cell in the probability image is either under or above the prob. value of interface  
    sgn = np.zeros_like(prob_img)  # sign function
    # initialize
    sgn[prob_img > zero_level] = 1
    sgn[prob_img < zero_level] = -1

    """ The euclidean distance transform gives values of the euclidean distance:
                      n
        y_i = sqrt(sum (x[i]-b[i])**2)
                      i
    where b[i] is the background point (value 0) with the smallest Euclidean distance to input points x[i],
    and n is the number of dimensions.
    """

    # the distance_tranform_edt returns the Euclidean distance of the inspected point from the nearest background point
    abs_dist = ndimage.distance_transform_edt(prob_img - zero_level)
    signed_dist_map = np.zeros_like(sgn)
    signed_dist_map[sgn == 1] = abs_dist[sgn == 1]
    signed_dist_map[sgn == -1] = -abs_dist[sgn == -1]      

    return signed_dist_map
    