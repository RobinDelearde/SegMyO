#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation of the PASCAL-VOC 2012 dataset with bounding box and labels priors with SegMyO
using a pre-trained model (SegMyO_segmentation) + selection with various criteria (SegMyO_selection)

author: Robin Deléarde @LIPADE (Université de Paris)
date: 23/12/2020
"""
#%% import libraries

import os
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms as T
from PIL import Image
import pickle

from SegMyO_segmentation import seg_images
from SegMyO_selection import compute_criteria_all_proposals, select_output
from extract_PASCAL_annotations import get_bbox_and_label, extract_items
from test_utils import fill_bbox, compute_iou_dice, fusion_seg_outputs, compute_iou_dice2

#%% inputs

dir_path = 'datasets/Pascal-VOC/2012/VOCtrainval_11-May-2012/'

image_dir = dir_path+'JPEGImages/'
img_list_path = dir_path+'ImageSets/Segmentation/val.txt'
with open(img_list_path, 'r') as f:
    img_list = f.read().splitlines()
f.close()

# Pascal_VOC labels (20 classes + background)
Pascal_VOC_labels = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike',
 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']
nb_classes = len(Pascal_VOC_labels)

#%% parameters

nb_images = len(img_list)
# nb_images = 20
train_set = True # whether grounth truth exists to compute scores (True for train/valid, False for test)
seg_model = 'Mask R-CNN' # choices: ['Mask R-CNN', 'Faster R-CNN', 'FCN', 'DeepLabV3']
fill_bbox_if_not_found = True # fill the bounding box when the expected item is not found
verbose = True

first_run = True
criterion_nr = 6 # criterion for the selection with SegMyO, from 0 to 7 in:
    # [seg_score, dist_to_edges_score, bbox_area_ratio, item_r_area, dist_to_center_score, semantic_score, my_score1, my_score2]
save_images = True

#%% define output directories

output_dir = 'seg_images/'

# seg_model
seg_model_output_dir = output_dir+seg_model+'/'
# with SegMyO selected criterion
SegMyO_seg_img_dir = seg_model_output_dir+'seg_images_SegMyO_criterion%d'%criterion_nr+'/'
SegMyO_seg_img_dir_all_items = seg_model_output_dir+'seg_images_all_items_SegMyO_criterion%d'%criterion_nr+'/'
SegMyO_output_file = seg_model_output_dir+'outputs_SegMyO_criterion%d'%criterion_nr+'.csv'
SegMyO_output_file2 = seg_model_output_dir+'all_scores_SegMyO_criterion%d'%criterion_nr+'.csv'

#%% prepare outputs (remove when using in script)

if first_run:
    os.mkdir(output_dir)
    # seg_model outputs and criteria scores
    os.mkdir(seg_model_output_dir)
    os.mkdir(seg_model_output_dir+'seg_outputs/')
    os.mkdir(seg_model_output_dir+'criteria_scores/')

# seg_model + SegMyO
os.mkdir(SegMyO_seg_img_dir)
os.mkdir(SegMyO_seg_img_dir_all_items)
with open(SegMyO_output_file,'w') as output_file:
      output_file.write('image_nr;item_nr;img_name;object;recognized_as;my_score;iou;dice\n')
output_file.close()
with open(SegMyO_output_file2,'w') as output_file:
    output_file.write('image_nr;img_name;class_'+';class_'.join(map(str,np.arange(nb_classes)))+'\n')
output_file.close()

#%% process dataset one image at a time

# for image_nr in range(nb_images):
def process(image_nr):
    print('processing image nr %d' %(image_nr+1))
    img_name = img_list[image_nr]
    img_path = image_dir+img_name+'.jpg'
    img = plt.imread(img_path)
    img_size = (img.shape[0], img.shape[1])
    # plt.imshow(img); plt.show()
    
    items, class_gt = extract_items(dir_path, img_name, train_set)
    
    masks_all_items_SegMyO = np.zeros((nb_classes, img.shape[0], img.shape[1]), dtype=np.uint8)
    
    for item_nr, item in enumerate(items):
        print('item nr %d' %(item_nr+1))
        
        # extract corresponding item in ground truth
        item_label, item_bbox = get_bbox_and_label(item, train_set, class_gt)
        
        if first_run:
            #############################
            # segmentation with seg_model + compute criteria scores for all proposals (1st run only)
            
            # perform segmentation
            img = plt.imread(img_path)
            image_tensor = T.functional.to_tensor(img) # shape [nb_lines, nb_cols, 3], normalize to 0-1
            seg_output = seg_images([image_tensor], seg_model)
            
            # save seg_outputs
            f = open(seg_model_output_dir+'seg_outputs/'+img_name+('_object%d' %item_nr)+'.bin', 'wb')
            pickle.dump(seg_output, f)
            f.close()
            
            # compute scores
            criteria_scores = compute_criteria_all_proposals(seg_model, seg_output, item_bbox, item_label, verbose)
            
            # save scores
            f = open(seg_model_output_dir+'criteria_scores/'+img_name+('_object%d' %item_nr)+'.bin', 'wb')
            pickle.dump(criteria_scores, f)
            f.close()
        
        else: # (next runs)
            #############################
            # segmentation with seg_model and SegMyO: load seg_output and criteria_scores
            with open(seg_model_output_dir+'seg_outputs/'+img_name+('_object%d' %item_nr)+'.bin', 'rb') as f:
                seg_output = pickle.load(f)
            with open(seg_model_output_dir+'criteria_scores/'+img_name+('_object%d' %item_nr)+'.bin', 'rb') as f:
                criteria_scores = pickle.load(f)
        
        #############################
        # segmentation with seg_model and SegMyO: select the best output with given scores_list as criterion
        if criteria_scores is not None:
            scores_list = criteria_scores[criterion_nr]
            # my_score1:
            # scores_list = (criteria_scores[0]+2*criteria_scores[1]+2*criteria_scores[2]+criteria_scores[3])/6
            # my_score2:
            # scores_list = np.min([criteria_scores[0],criteria_scores[1],criteria_scores[2]], axis=0)
            # C3*semantic score:
            # scores_list = criteria_scores[3]*criteria_scores[5]
        mask, predicted_label, my_score = \
            select_output(img_size, seg_model, seg_output, scores_list, item_bbox, item_label, verbose=verbose)
        
        # fill item_bbox if object not found
        if predicted_label is None and fill_bbox_if_not_found:            
            mask = fill_bbox(img_size, item_bbox)
        
        if train_set:
            # compute segmentation scores
            iou_score, dice_score = compute_iou_dice(mask, item)
        else:
            iou_score, dice_score = 0., 0. # no ground truth available for test set
        
        # write result in output file
        with open(SegMyO_output_file,'a') as output_file:
            output_file.write(str(image_nr)+';'+str(item_nr)+';'+img_name+';'+ \
                str(item_label)+';'+str(predicted_label)+';'+'{:.2f}'.format(my_score)+';'+\
                '{:.2f}'.format(iou_score)+";"+'{:.2f}'.format(dice_score)+'\n')
        output_file.close()
        
        # save output image
        if save_images:
            im_pil = Image.fromarray(255*mask)
            im_pil.save(SegMyO_seg_img_dir+img_name+'_object%d_'%item_nr+item_label+'.png')
        
        # add to image containing all items
        masks_all_items_SegMyO[Pascal_VOC_labels.index(item_label)] += mask
    
    #############################
    # fusion of segmentation outputs into 1 image with all items + compute IoU for each class
    
    img_all_items_SegMyO = fusion_seg_outputs(masks_all_items_SegMyO)
    if save_images:
        im_pil_SegMyO = (Image.fromarray(img_all_items_SegMyO)).convert('RGB')
        im_pil_SegMyO.save(SegMyO_seg_img_dir_all_items+img_name+'_segmented.png')
    iou_scores, dice_scores = compute_iou_dice2(img_all_items_SegMyO, class_gt, nb_classes)
    with open(SegMyO_output_file2,'a') as output_file:
        output_file.write(str(image_nr)+';'+img_name+';'+(';'.join(map(str,iou_scores)))+'\n')
    output_file.close()

#%% run process in a python IDE

for image_nr in range(nb_images):
    process(image_nr)

#%% run process in a script
# args: image_nr

import sys
from distutils.version import LooseVersion
import torch

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'
    process(int(sys.argv[1]))
