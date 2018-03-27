# -*- coding: utf-8 -*-
"""
Created on Sat Mar  11 11:20:14 2018

@author: plin@augusta.edu
"""

import numpy as np

def Bresenham_line(ends):
    w, h = np.diff(ends, axis=0)[0]
    pt1, __ = ends
    x, y = pt1

    longest = np.absolute(w)
    shortest = np.absolute(h)    
    
    if w != 0:
        dx = int(np.absolute(w)/w)
    if h != 0:
        dy = int(np.absolute(h)/h)
    
    if not (longest > shortest):
        longest = np.absolute(h)
        shortest = np.absolute(w)
    
    numerator = int(longest/2)
    for i in range(longest+1):
        yield x, y
        numerator += shortest
        if not (numerator < longest):
            numerator -= longest
            x += dx
            y += dy
        elif (np.absolute(w) > np.absolute(h)):
            x += dx
        else: 
            y += dy       


def connect(ends):
    w, h = np.abs(np.diff(ends, axis=0))[0]
    if w > h: 
        return np.c_[np.linspace(ends[0, 0], ends[1, 0], w+1, dtype=np.int32),
                     np.round(np.linspace(ends[0, 1], ends[1, 1], w+1))
                     .astype(np.int32)]
    else:
        return np.c_[np.round(np.linspace(ends[0, 0], ends[1, 0], h+1))
                     .astype(np.int32),
                     np.linspace(ends[0, 1], ends[1, 1], h+1, dtype=np.int32)]


def Murphy_line_vectorize(start_pts, end_pts, thk = 1):
     # Modified verion of Murphy's 2D line thickening through vectorization
    w, h = np.abs(start_pts - end_pts)[0]
        
    if w > h:
        return np.c_[(np.linspace(start_pts[0, 0], end_pts[0, 0], w+1, dtype=np.int32)[:,None] + start_pts[:,0] - start_pts[0, 0]).reshape(-1), \
                np.round(np.linspace(start_pts[0, 1], end_pts[0, 1], w+1, dtype=np.int32)[:,None] + start_pts[:,1] - start_pts[0, 1]).reshape(-1)]
    else:
        return np.c_[np.round(np.linspace(start_pts[0, 0], end_pts[0, 0], h+1, dtype=np.int32)[:,None] + start_pts[:,0] - start_pts[0, 0]).reshape(-1), \
                (np.linspace(start_pts[0, 1], end_pts[0, 1], h+1, dtype=np.int32)[:,None] + start_pts[:,1] - start_pts[0, 1]).reshape(-1)]

    
    
def connect2(ends):
    d0, d1 = np.diff(ends, axis=0)[0]
    if np.abs(d0) > np.abs(d1): 
        return np.c_[np.arange(ends[0, 0], ends[1,0] + np.sign(d0), np.sign(d0), dtype=np.int32),
                     np.arange(ends[0, 1] * np.abs(d0) + np.abs(d0)//2,
                               ends[0, 1] * np.abs(d0) + np.abs(d0)//2 + (np.abs(d0)+1) * d1, d1, dtype=np.int32) // np.abs(d0)]
    else:
        return np.c_[np.arange(ends[0, 0] * np.abs(d1) + np.abs(d1)//2,
                               ends[0, 0] * np.abs(d1) + np.abs(d1)//2 + (np.abs(d1)+1) * d0, d0, dtype=np.int32) // np.abs(d1),
                     np.arange(ends[0, 1], ends[1,1] + np.sign(d1), np.sign(d1), dtype=np.int32)]


def connect_nd(ends):
    d = np.diff(ends, axis=0)[0]
    j = np.argmax(np.abs(d))
    D = d[j]
    aD = np.abs(D)
    return ends[0] + (np.outer(np.arange(aD + 1), d) + (aD>>1)) // aD
