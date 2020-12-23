#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation utilities

author: Robin Deléarde @LIPADE (Université de Paris)
date: 23/12/2020
"""

import numpy as np

def draw_bbox(img, bbox_list, colors_list=None): # 3 channel image
    # if colors_list is None:
    #     colors_list = #TODO: set random colors
    img_bbox = np.copy(img)
    for i, bbox in enumerate(bbox_list):
        img_bbox[bbox[0]:bbox[1], bbox[2]] = colors_list[i]
        img_bbox[bbox[0]:bbox[1], bbox[3]] = colors_list[i]
        img_bbox[bbox[0], bbox[2]:bbox[3]] = colors_list[i]
        img_bbox[bbox[1], bbox[2]:bbox[3]] = colors_list[i]
    return img_bbox

def fill_bbox(img_size, bbox=None, coeff=0.9):
    img_full = np.zeros(img_size, dtype=np.uint8)
    if bbox is not None:
        height = int((bbox[1]-bbox[0])*coeff)
        width = int((bbox[3]-bbox[2])*coeff)
        box_x_left = int((bbox[1]-bbox[0])*(1.-coeff))
        box_y_left = int((bbox[3]-bbox[2])*(1.-coeff))
        img_seg = np.ones((height, width), dtype=np.uint8)
        img_full[bbox[0]+box_x_left//2:bbox[0]+box_x_left//2+height, bbox[2]+box_y_left//2:bbox[2]+box_y_left//2+width] = img_seg
    return img_full

def compute_iou_dice(mask_pred, mask_gt):
    area_gt = np.sum(mask_gt)
    area_pred = np.sum(mask_pred)
    area_intersect = np.sum(np.logical_and(mask_gt, mask_pred))
    # area_union = np.sum(np.logical_or(mask_gt, mask_pred))
    iou_score = area_intersect / (area_gt+area_pred-area_intersect)
    dice_score = 2.*area_intersect / (area_gt+area_pred)
    return iou_score, dice_score

def fusion_seg_outputs(masks_all_items):
    """
    choose the best class for each pixel by taking the smallest one in case of conflict (overlap)
    and compute scores globaly for each class (1 class = 1 instance)
    
    masks_all_items: binary masks
    """
    # nb_classes = masks_all_items.shape[0]
    img_size = (masks_all_items.shape[1], masks_all_items.shape[2])
    
    # initialize outputs
    img_seg_full = np.zeros(img_size, dtype=np.uint8)
    
    class_areas = masks_all_items.sum(axis=1).sum(axis=1)
    order = np.argsort(class_areas)[::-1] # sort in descending order
    
    for i in order:
        mask = masks_all_items[i]
        x,y = np.where(mask>0)
        img_seg_full[x,y] = i
    
    return img_seg_full

def compute_iou_dice2(img_seg_full, img_gt, nb_classes):
    iou_scores = np.zeros(nb_classes)
    dice_scores = np.zeros(nb_classes)
    
    for i in range(nb_classes):
        mask_pred = (img_seg_full==i)
        mask_gt = (img_gt==i)
        iou_scores[i], dice_scores[i] = compute_iou_dice(mask_pred, mask_gt)
    
    return iou_scores, dice_scores
