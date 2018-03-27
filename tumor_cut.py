# -*- coding: utf-8 -*-
"""
Created on Sat Mar  13 11:37:00 2018

@author: Power-R7
"""
import numpy as np
from numpy.linalg import norm as nrm
from scipy import ndimage
from signed_distance_function import *
import matplotlib.pyplot as plt

class tumorCut(object):
    
    def __init__(self, image, foreground_seeds, background_seeds):
        """
        Construct a new tumorCut instance. 
        Inputs:
        - image: A 2D image of 1 (gray scale) or 3 (color) scale generated by contrast-enhanced T1W MRI.
        - foreground_seeds: A 2D array containing (x, y) coordinates used to specify th tumor seeds.
        - background_seeds: A 2D array containing (x, y) coordinates used to delineate the bounding box enclosing
                            the tumor volume. 
        """
        self.img = image
        self.fg_seeds = foreground_seeds
        self.bg_seeds = background_seeds
        self.params = {}
        # prarmeter initialization
        self.params['crop_label_fg'] = []
        self.params['crop_label_bg'] = []
        self.params['crop_img'] = []
        self.params['s_map'] = []
        
        
    def cropImage(self, img = []):
        """ Crop image using the bounding box defined by the background seeds.
            The cropped image is further used for image segmentation
        """
        if img == []:
            img = self.img
        fg_seeds = self.fg_seeds.astype(int)
        bg_seeds = self.bg_seeds.astype(int)
        # Specify the coordinates of the vertecies of the bounding box
        # bg_seeds.max(axis=0), bg_seeds.min(axis=0) = [115.,  83.] [107.,  60.]
        # crop image by index slicing
        crop_image = img[bg_seeds.min(axis=0)[1]:bg_seeds.max(axis=0)[1]+1, bg_seeds.min(axis=0)[0]:bg_seeds.max(axis=0)[0]+1]
        shiftFg_seeds = fg_seeds - bg_seeds.min(axis=0)
        shiftBg_seeds = bg_seeds - bg_seeds.min(axis=0)
        self.params['crop_img'] = crop_image
        
        return crop_image, shiftFg_seeds, shiftBg_seeds 
  
      
    def selectVOI(self, CA_state = 'fg'):
        """generate a matrix of size equal to the size of loaded image, and then label the seeded matrix elements 
            as background (or foreground) seeds, and then crop the image before using Grow-cut algorithm for image segmentation 
        """
        # CA_state: indicate what set of seeds is used for running the tumor CA algorithm; the value is either "fg", foreground seeds
        #           or "bg", background seeds.
        img = self.img
        
        if len(img.shape) == 2:
            label_matrix = np.zeros_like(img)
        else:
            label_matrix = np.zeros_like(img[:,:,0])

        #label_matrix = np.zeros_like(img)
        crop_label, shift_fg, shift_bg = self.cropImage(label_matrix)
        
        if CA_state == 'fg':
            crop_label[shift_fg[:,1], shift_fg[:,0]] = 1
            self.params['crop_label_fg'] = crop_label
            return crop_label
        elif CA_state == 'bg':
            crop_label[shift_bg[:,1], shift_bg[:,0]] = 1
            self.params['crop_label_bg'] = crop_label
            return crop_label

    
    def Grow_Cut(self, seeds = 'fg', beta = 1.0, diff = 10):
        """Crow-cut method uses a continuous state cellular automata to interacticely label images 
           using supplied seeds. The cells refer to image pixels.
           The automata is initialized by assigning corresponding labels ast seeds with a strength value
           between 0 and 1 where the higher value reflects a higher confidence in choosing the seed.  Strengths
           of unlabeled cells are set to 0.
        """
        # need crop_label from selectVOI() and crop_img from cropImage()
        # start the CA-based method by using the foreground/ backgroud seeds
        if seeds == 'fg':
            crop_label = self.selectVOI('fg').astype(float)
        elif seeds == 'bg':
            crop_label = self.selectVOI('bg').astype(float)
        crop_img = self.cropImage()[0].astype(int)/255.
        
        img_pad = np.pad(crop_img, (1,1), 'constant')
        label_pad = np.pad(crop_label, (1,1), 'constant')
        x = crop_label.copy()
        x_pad = np.pad(x, (1,1), 'constant')
        
        # Check dimensions
        assert (crop_label.shape == crop_img.shape), 'dimensions of cropped images do not match'
        
        while diff > 1e-4: 
            for i in range(crop_label.shape[0]):
                for j in range(crop_label.shape[1]):
                    a =  np.exp(-np.absolute(crop_img[i,j]-img_pad[i:i+3,j:j+3])*(1-(1-beta)*label_pad[i:i+3,j:j+3]*(crop_img[i,j]>img_pad[i:i+3,j:j+3]).astype(float)))
                    a = np.multiply(a, x_pad[i:i+3,j:j+3])
 #                   a[1,1] = 0
                    x[i,j] = a.max()
                    crop_label[i,j] = label_pad[tuple(np.add(np.unravel_index(np.argmax(a, axis=None), a.shape), (i,j)))]
            diff = nrm(label_pad[1:-1,1:-1] - crop_label)/np.multiply(crop_label.shape[0], crop_label.shape[1])
            label_pad[1:-1,1:-1] = crop_label
            x_pad[1:-1,1:-1] = x
        
        return crop_label, x
    
    
    def ProbMap(self, fg_beta = 0.7, bg_beta = 1.0):
        # combine foreground and background strength maps to generate tumor probability map
        __, fg_x = self.Grow_Cut('fg', fg_beta)
        __, bg_x = self.Grow_Cut('bg', bg_beta)
        
        return np.log(bg_x)/(np.log(bg_x)+np.log(fg_x))
    
    
    def Level_Set_Evolution(self, zero_level = 0.5, dt = 0.1):
        # The level set function with zero-level on the initial estimate of the tumor surface (obtain from ProbMap())
        # is evolved on the probability map with a piecewise constant region assumption, locally smoothed by using a
        # Gaussian kernal to define inner and outer regions around the propagation surface. 
        
        #prob_img = ndimage.filters.gaussian_filter(self.ProbMap(), 0.25, mode='nearest')
        prob_img = np.around(self.ProbMap(), decimals=1)
        s_map = sdf(prob_img, zero_level)
        S_diff = None
        while S_diff is None or S_diff > 1e-3:
            u = np.mean(prob_img[s_map > 0.])  # mean of the region inside the current estimated tumor surface
            v = np.mean(prob_img[s_map < 0.])  # mean of the region inside the current estimated tumor surface
            dsdt = (u - v)*(u + v - 2*prob_img)
            preS = s_map.copy()
            s_map += dt*dsdt
            s_map = ndimage.filters.gaussian_filter(s_map, 0.3, mode='nearest')
            S_diff = nrm(preS-s_map)/(s_map.shape[0]*s_map.shape[1])
            self.params['s_map'] = s_map
            
        return S_diff, s_map
    
    def SegMap_on_Img(self):
        # Check dimensions of the cropped image and the segmented image
        crop_img = self.params['crop_img']
        s_map = self.params['s_map']
        bg = self.bg_seeds.astype(int)
        assert (s_map.shape == crop_img.shape), 'dimensions of cropped image and segmented image do not match'
        upperleft_coord = bg.min(axis=0)
        # generate an image template with an original image size and attach the sdf map on it.
        img_tmp = np.zeros_like(self.img).astype(float)
        img_tmp[upperleft_coord[1]:upperleft_coord[1]+s_map.shape[1], upperleft_coord[0]:upperleft_coord[0]+s_map.shape[0]] = s_map
        
        return img_tmp
        