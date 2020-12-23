#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation of an image using bounding box and labels priors with SegMyO,
using a pre-trained model (SegMyO_segmentation) + selection with various criteria (SegMyO_selection)

author: Robin Deléarde @LIPADE (Université de Paris)
date: 23/12/2020
"""
#%% import libraries

import os
import numpy as np
from matplotlib import pyplot as plt
from torch import from_numpy
# from torchvision import transforms as T
from PIL import Image
import pickle

from SegMyO_segmentation import seg_images
from SegMyO_selection import compute_criteria_all_proposals, select_output

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

#%% inputs

image_dir = ''
# image_dir = 'datasets/Pascal-VOC/2012/VOCtrainval_11-May-2012/JPEGImages/'
img_name = '2007_000033'
output_dir = 'SegMyO_outputs/'+img_name+'/'

# load data and prepare outputs
img_path = image_dir+img_name+'.jpg'
img = plt.imread(img_path)
img_size = img.shape[0:2]
plt.imshow(img); plt.show()
os.makedirs(output_dir)

#%% 1. pre-segmentation

# parameters
seg_model = 'Mask R-CNN' # choices: ['Mask R-CNN', 'Faster R-CNN', 'FCN', 'DeepLabV3']

# perform segmentation
tensor_image = from_numpy(img.transpose((2, 0, 1)))/255. # shape [3, nb_lines, nb_cols], normalize to [0,1]
# tensor_image = T.functional.to_tensor(img) # shape [3, nb_lines, nb_cols], normalize to [0,1]
seg_output = seg_images([tensor_image], seg_model)

# save seg_outputs
f = open(output_dir+'seg_outputs.bin', 'wb')
pickle.dump(seg_output, f)
f.close()

#%% 2. compute criteria scores

# inputs
item_nr = 1
item_label = 'aeroplane'
item_bbox = [109, 261, 8, 499]
# item_nr = 2
# item_bbox = [188, 222, 325, 411]*
# item_nr = 3
# item_bbox = [200, 222, 419, 482]

# save original image with bbox
if img.max()>1:
    img_bbox = draw_bbox(img, [item_bbox], [(255, 0, 0)])
    im_pil = Image.fromarray(img_bbox)
else:
    img_bbox = draw_bbox(img, [item_bbox], [(1., 0., 0.)])
    im_pil = Image.fromarray((255*img_bbox).astype(np.uint8))
plt.imshow(img_bbox); plt.show()
im_pil.save(output_dir+img_name+('_object%d_input_bbox.png' %item_nr))

# load segmentation outputs computed previously
with open(output_dir+'seg_outputs.bin', 'rb') as f:
    seg_output = pickle.load(f)

# compute scores
criteria_scores = compute_criteria_all_proposals(seg_model, seg_output, item_bbox, item_label)

# save scores
os.mkdir(output_dir+'criteria_scores/')
f = open(output_dir+'criteria_scores/object%d.bin' %item_nr, 'wb')
pickle.dump(criteria_scores, f)
f.close()

#%% 3. select best output with given criterion

# parameters
criterion_nr = 6 # from 0 to 7 in [seg_score, dist_to_edges_score, bbox_area_ratio, item_r_area, dist_to_center_score, semantic_score, my_score1, my_score2]
score_thresh = None # threshold for the score
bin_thresh = None # binarization threshold for the output mask
remove_outside_bbox = True # remove the content outside of the bounding box from the output mask

# load criteria scores computed previously
with open(output_dir+'criteria_scores/object%d.bin' %item_nr, 'rb') as f:
    seg_scores = pickle.load(f)

# perform selection
scores_list = criteria_scores[criterion_nr]
mask, predicted_label, my_score = \
    select_output(img_size, seg_model, seg_output, scores_list, item_bbox, item_label, score_thresh, bin_thresh, remove_outside_bbox)
plt.imshow(mask); plt.show()

# save output mask
mask_pil = Image.fromarray((255*mask).astype(np.uint8))
mask_pil.save(output_dir+img_name+('_object%d_output_mask.png' %item_nr))
