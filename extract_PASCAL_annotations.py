#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract label and bbox from PASCAL VOC annotations

author: Robin Deléarde @LIPADE
date: 09/2020
"""

import numpy as np
from PIL import Image
from xml.dom import minidom
from extract_seg_masks import extract_seg_masks, get_class_and_bbox

#%% Functions: extract label and bbox from PASCAL VOC annotations

# Pascal_VOC labels (20 classes + background)
Pascal_VOC_labels = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike',
 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']

def convert_PASCALVOC_annot(label):
    if label == 'diningtable':
        return 'dining table'
    elif label == 'pottedplant':
        return 'potted plant'
    elif label == 'tvmonitor':
        return 'tv monitor'
    else:
        return label

# Extract label and bbox from PASCAL VOC annotations
def get_bbox_and_label(item, train_set, class_gt):
    if train_set:
        # bounding box and label from segmentation ground truth (for train_set only, easier to compute the score,
        # otherwise need to match the object from bbox annotations with the object from the segmentation annotations for each object)
        object_class, item_bbox = get_class_and_bbox(item, class_gt)
        item_label = Pascal_VOC_labels[object_class]
    else:
        # with bounding box and label annotations (for test)
        object_name = item.getElementsByTagName('name')[0].firstChild.data # item.attributes['name'].value doesn't work
        item_label = convert_PASCALVOC_annot(object_name)
        item_bbox_dom = item.getElementsByTagName('bndbox')[0]
        item_bbox_xmin = int(item_bbox_dom.getElementsByTagName('xmin')[0].firstChild.data)-1
        item_bbox_ymin = int(item_bbox_dom.getElementsByTagName('ymin')[0].firstChild.data)-1
        item_bbox_xmax = int(item_bbox_dom.getElementsByTagName('xmax')[0].firstChild.data)-1
        item_bbox_ymax = int(item_bbox_dom.getElementsByTagName('ymax')[0].firstChild.data)-1
        item_bbox = np.array([item_bbox_ymin, item_bbox_ymax, item_bbox_xmin, item_bbox_xmax], dtype=np.uint8)
    return item_label, item_bbox

# Extract items for a given object
def extract_items(dir_path, img_name, train_set):
    if train_set:
        # extract ground truth masks
        seg_gt_pil = Image.open(dir_path+'SegmentationObject/'+img_name+'.png')
        seg_gt = np.array(seg_gt_pil)
        items, background, borders = extract_seg_masks(seg_gt)
        class_gt_pil = Image.open(dir_path+'SegmentationClass/'+img_name+'.png')
        class_gt = np.array(class_gt_pil)
    else:
        # extract bounding box and labels annotations
        annot_file = dir_path+'Annotations/'+img_name+'.xml'
        mydoc = minidom.parse(annot_file)
        items = mydoc.getElementsByTagName('object')
        class_gt = None
    return items, class_gt
