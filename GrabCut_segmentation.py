#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GrabCut segmentation

see: https://www.pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/

author: Robin Deléarde @LIPADE
date: 28/09/2020, last modified on the 12/10/2020
"""

import numpy as np
import cv2
# from matplotlib import pyplot as plt

def GrabCut_segmentation(img, bbox, iterCount=5):
    mask_init = np.zeros(img.shape[:2],np.uint8) # initial approximate mask (for GC_INIT_WITH_MASK mode)
    bgdModel = np.zeros((1,65),np.float64) # temporary array used by GrabCut internally when modeling the background
    fgdModel = np.zeros((1,65),np.float64) # temporary array used by GrabCut internally when modeling the foreground
    
    rect = (bbox[2],bbox[0],bbox[3]-bbox[2],bbox[1]-bbox[0])
    # rect = (y_min, x_min, width, height)
    # from bbox = [x_min, x_max, y_min, y_max]
    
    mask_full_img, bgdModel, fgdModel = cv2.grabCut(img, mask_init, rect, bgdModel, fgdModel, iterCount, cv2.GC_INIT_WITH_RECT)
    # mask values: 0 = Definite background, 1 = Definite foreground, 2 = Probable background, 3 = Probable foreground
    mask_full_img = (mask_full_img==3).astype('uint8')
    mask = mask_full_img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    
    return mask, mask_full_img

def GrabCut_segmentation_with_mask(img, mask_init, iterCount=5):
    bgdModel = np.zeros((1,65),np.float64) # temporary array used by GrabCut internally when modeling the background
    fgdModel = np.zeros((1,65),np.float64) # temporary array used by GrabCut internally when modeling the foreground
    
    mask_full_img, bgdModel, fgdModel = cv2.grabCut(img, mask_init, None, bgdModel, fgdModel, iterCount, cv2.GC_INIT_WITH_MASK)
    # mask values: 0 = Definite background, 1 = Definite foreground, 2 = Probable background, 3 = Probable foreground
    mask_full_img = (mask_full_img==1).astype('uint8')
    
    return mask_full_img
