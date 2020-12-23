#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select the best segment in a list of proposals (the output of a segmentation model),
for a given bounding box and/or a given label, by using a combination of several
geometric criteria on the bounding box (or the image if no bounding box is given),
and a semantic criterion if the expected label is given.

based on segmentation models from:
    - https://pytorch.org/docs/stable/torchvision/models.html

author: Robin Deléarde @LIPADE (Université de Paris)
date: 23/12/2020
"""
#%% import libraries

import numpy as np
#from skimage.morphology import area_opening # used to remove small regions in the segmentation outputs (optional)
from sematch.semantic.similarity import WordNetSimilarity # WordNet similarity with sematch (https://github.com/gsi-upm/sematch)
from test_utils import compute_iou_dice

wns = WordNetSimilarity()

#%% labels for pytorch/torchvision models trained on COCO or PASCAL VOC (https://pytorch.org/docs/stable/torchvision/models.html)

# labels for torchvision Mask R-CNN and Faster R-CNN {detection + instance segmentation} models, pre-trained on COCO (80 classes)
COCO_labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# labels for torchvision FCN and DeepLabV3 segmentation models, pre-trained on COCO subset with Pascal_VOC classes (20 classes)
Pascal_VOC_labels = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike',
 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']

#%% compute criteria

def compute_criteria(seg_model, seg_output, proposal_nr, item_bbox=None, item_label=None):
    """
    Compute the criteria scores for a given bounding box and/or a given label,
    for a given region/segment proposal in a segmentation output.
    
    args:
    seg_output: segmentation output dictionary structure with attributes:
        'labels': list of predicted labels
        'scores': list of segmentation scores
        'masks': list of segmentation masks
    item_label: label of the expected object (if None, use geometric criteria only)
    item_bbox: bbox coordinates of the expected object in the original image, in format 
        array([x_start, x_stop, y_start, y_stop]), None will take the full image
    
    output:
    1D array containing the scores for each criterion, in the following order:
    [seg_score, dist_to_edges_score, bbox_area_ratio, item_r_area, dist_to_center_score, semantic_score, my_score1, my_score2]
    """
    current_item = seg_output['labels'][proposal_nr]
    seg_score = seg_output['scores'][proposal_nr] # criterion 0: segmentation score
    mask = seg_output['masks'][proposal_nr]
    if seg_model in ['Mask R-CNN', 'Faster R-CNN', 'FCN', 'DeepLabV3']:
        seg_score = seg_score.item()
        mask = mask[0].detach().numpy()
        if seg_model in ['Mask R-CNN', 'Faster R-CNN']:
            current_item = COCO_labels[current_item.item()]
        elif seg_model in ['FCN', 'DeepLabV3']:
            current_item = Pascal_VOC_labels[current_item.item()]
    else:
        raise Exception('segmentation model not implemented')
    
    # plt.imshow(mask); plt.show()
    # mask = area_opening(mask, max(400, mask.shape[0]*mask.shape[1]//135)) # remove small regions (optional)
    # plt.imshow(mask); plt.show()
    
    mask_area = mask.sum()
    if item_bbox is None:
        mask_bbox = mask
        item_bbox = np.array([0, mask.shape[0], 0, mask.shape[1]], dtype=np.uint8)
    else:
        mask_bbox = mask[item_bbox[0]:item_bbox[1], item_bbox[2]:item_bbox[3]]
    mask_bbox_area = mask_bbox.sum()
    
    if mask_bbox_area==0:
        return np.zeros((8)) # set all scores to 0 if item is oustide bbox
    
    else:
        # geometric criterion 1: distance to the edges of the bbox -> basic necessary and universal criterion
        x_bbox,y_bbox = np.nonzero(mask_bbox)
        if len(x_bbox)>0: # equivalent to mask_bbox_area>0
            dist_up = min(x_bbox)/mask_bbox.shape[0]
            dist_down = (mask_bbox.shape[0]-1-max(x_bbox))/mask_bbox.shape[0]
            dist_left = min(y_bbox)/mask_bbox.shape[1]
            dist_right = (mask_bbox.shape[1]-1-max(y_bbox))/mask_bbox.shape[1]
            max_distance = max(dist_up, dist_down, dist_left, dist_right)
        else:
            max_distance = 1.
        dist_to_edges_score = 1.-max_distance
        
        # geometric criterion 2: bbox_area_ratio (NB: bad if the object is not a single instance)
        bbox_area_ratio = mask_bbox_area/mask_area
        
        # geometric criterion 3: item relative area -> to put more weight to large objects (relatively to the bbox size)
        item_r_area = mask_bbox_area/mask_bbox.shape[0]/mask_bbox.shape[1]
        
        # geometric criterion 4: distance to the center of the bounding box -> only for barycentric objects or to avoid objects only on the edges
        center_x, center_y = item_bbox[0]+mask_bbox.shape[0]//2, item_bbox[2]+mask_bbox.shape[1]//2 # center of bbox/image
        x,y = np.nonzero(mask)
        # dist = np.sqrt((x-center_x)**2+(y-center_y)**2)
        dist = abs(x-center_x)/mask_bbox.shape[0]+abs(y-center_y)/mask_bbox.shape[1]
        dist_norm = 1-np.exp(-dist)
        mask_values = mask[x,y]
        dist_to_center_score = 1.-(mask_values*dist_norm).sum()/mask_area # NB: mask_bbox_area=mask_values.sum()
        
        # semantic criterion: label matching
        if item_label is not None and current_item is not None:
            if current_item.replace(' ', '_')==item_label.replace(' ', '_'):
                semantic_score = 1.
            else:
                semantic_score = wns.word_similarity(current_item.replace(' ', '_'), item_label.replace(' ', '_'), 'wpath')
                # 3th argument = method, among: 'li', 'lin', 'wup', 'res', 'jcn', 'wpath' (cf. https://gsi-upm.github.io/sematch/similarity/)
        
        # global criterion
        if item_label is not None and semantic_score>0:
            my_score1 = (seg_score + 2*bbox_area_ratio + 2*dist_to_edges_score + item_r_area + dist_to_center_score + 5*semantic_score)/12
        else:
            my_score1 = (seg_score + 2*bbox_area_ratio + 2*dist_to_edges_score + item_r_area + dist_to_center_score)/7
        if seg_model in ['Mask R-CNN', 'Faster R-CNN']: # instance segmentation => bbox_area_ratio should be close to 1
            my_score2 = min(seg_score, bbox_area_ratio, dist_to_edges_score, my_score1)
        else:
            my_score2 = min(seg_score, dist_to_edges_score, my_score1)
    
    return np.array([seg_score, dist_to_edges_score, bbox_area_ratio, item_r_area, dist_to_center_score, semantic_score, my_score1, my_score2])

def compute_criteria_all_proposals(seg_model, seg_output, item_bbox=None, item_label=None, verbose=True):
    """
    Compute the criteria scores for a given bounding box and/or a given label,
    for all region/segment proposals in a segmentation output.
    
    args:
    seg_output: segmentation outputs dictionary structure with attributes:
        'labels': list of predicted labels
        'scores': list of segmentation scores
        'masks': list of segmentation masks
    item_bbox: bbox coordinates of the expected object in the original image, in format 
        array([x_start, x_stop, y_start, y_stop]), None will take the full image
    item_label: label of the expected object (if None, use geometric criteria only)
    
    output:
    2D array containing the scores for each proposal (lines) and each criterion (columns), in the following order:
    [seg_score, dist_to_edges_score, bbox_area_ratio, item_r_area, dist_to_center_score, semantic_score, my_score1, my_score2])
    """
    nb_items = len(seg_output['scores'])
    if nb_items==0:
        if verbose:
            print('Did not find any item in segmentation outputs')
        return None
    else:
        scores = np.zeros((nb_items, 8))
        for i in range(nb_items):
            scores[i] = compute_criteria(seg_model, seg_output, i, item_bbox, item_label)
        return np.transpose(scores)

#%% select the best output

def select_output(img_size, seg_model, seg_output, scores_list, item_bbox=None, item_label=None, score_thresh=0.25, bin_thresh=0.3, remove_outside_bbox=True, verbose=True):
    """
    Select the best region/segment proposal in a segmentation output
    for a given bounding box and/or a given label,
    based on a list of criteria scores computed previously.
    
    args:
    seg_output: segmentation outputs dictionary structure with attributes:
        'labels': list of predicted labels
        'scores': list of segmentation scores
        'masks': list of segmentation masks
    scores_list: 1D array of scores for all proposals (1 score/proposal)
    item_bbox: bbox coordinates of the expected object in the original image, in format 
        array([x_start, x_stop, y_start, y_stop]), None will take the full image
    item_label: label of the expected object (if None, use geometric criteria only)
    score_thresh: threshold for the score, return None output if score<score_thresh for all items
    bin_thresh: threshold for the binarization of the segmentation mask (no binarization if None)
    
    outputs:
    mask: mask of the selected output (binary mask if bin_thresh is not None)
    predicted_label: label of the selected output
    my_score: score of the selected output
    """
    nb_items = len(seg_output['scores'])
    if nb_items==0:
        if verbose:
            print('Did not find any item in segmentation outputs')
        empty_img = np.zeros(img_size, dtype=np.uint8)
        return empty_img, None, 0.
    else:
        # # uncomment to compute scores inside selection (NB: scores are not returned here)
        # criteria_scores = compute_criteria_all_proposals(seg_model, seg_output, item_bbox, item_label, verbose)
        # scores_list = criteria_scores[criterion_nr]
        
        my_score = np.max(scores_list)
        i = np.argmax(scores_list)
        
        if score_thresh is not None and my_score<score_thresh:
            if verbose:
                print('Did not find any item with score > score_tresh')
            empty_img = np.zeros(img_size, dtype=np.uint8)
            return empty_img, None, 0.
        else:
            if seg_model in ['Mask R-CNN', 'Faster R-CNN']:
                predicted_label = COCO_labels[seg_output['labels'][i].item()]
            elif seg_model in ['FCN', 'DeepLabV3']:
                predicted_label = Pascal_VOC_labels[seg_output['labels'][i].item()]
            
            if verbose:
                print('Found item "'+str(predicted_label)+'" with score %.2f' %my_score)
                
            # extract mask image
            mask = seg_output['masks'][i]
            if seg_model in ['Mask R-CNN', 'Faster R-CNN', 'FCN', 'DeepLabV3']:
                mask = mask[0].detach().numpy()
            # plt.imshow(mask); plt.show()
            if bin_thresh is not None:
                mask = (mask>=bin_thresh).astype(np.uint8) # binarization of the segmentation
                # plt.imshow(img_seg); plt.show()
            
            # remove part outside of the bounding box (optional)
            if remove_outside_bbox and item_bbox is not None:
                mask_bbox = mask[item_bbox[0]:item_bbox[1], item_bbox[2]:item_bbox[3]]
                if bin_thresh is not None:
                    mask = np.zeros(img_size, dtype=np.uint8)
                else:
                    mask = np.zeros(img_size, dtype=np.float32)
                mask[item_bbox[0]:item_bbox[1], item_bbox[2]:item_bbox[3]] = mask_bbox
            # plt.imshow(mask); plt.show()
                
            return mask, predicted_label, my_score

#%% compute mIoU for each mask with given ground truth

def compute_mIoUs(seg_output, seg_model, mask_gt, item_bbox, seg_thresh=0.25, bin_thresh=0.3, verbose=True):
    """
    compute mIoU for each mask in seg_output with seg_score>seg_thresh, with mask_gt as ground truth
    only with bbox_before = False
    """
    nb_items = len(seg_output['scores'])
    if nb_items==0:
        if verbose:
            print('Did not find any item in segmentation outputs')
        return None
    else:
        seg_score = seg_output['scores'][0]
        if seg_model in ['Mask R-CNN', 'Faster R-CNN', 'FCN', 'DeepLabV3']:
            seg_score = seg_score.item()
        if seg_score<seg_thresh:
            if verbose:
                print('Segmentation score < threshold for all items')
            return None
        else:
            i=0
            mIoU_list = []
            while i<nb_items and seg_score>=seg_thresh:
                seg_score = seg_output['scores'][i] # criterion 1: segmentation score (in [0,1])
                if seg_model in ['Mask R-CNN', 'Faster R-CNN', 'FCN', 'DeepLabV3']:
                    seg_score = seg_score.item()
                if seg_score>=seg_thresh: # necessary for last item in while loop
                    mask = seg_output['masks'][i]
                    if seg_model in ['Mask R-CNN', 'Faster R-CNN', 'FCN', 'DeepLabV3']:
                        mask = mask[0].detach().numpy()
                    mask = (mask>=bin_thresh).astype(np.uint8) # binarization of the segmentation
                    img_seg = mask[item_bbox[0]:item_bbox[1], item_bbox[2]:item_bbox[3]]
                    img_full = np.zeros(mask.shape)
                    img_full[item_bbox[0]:item_bbox[1], item_bbox[2]:item_bbox[3]] = img_seg
                    iou_score, dice_score = compute_iou_dice(img_full, mask_gt)
                    mIoU_list.append(iou_score)
                i+=1
            return mIoU_list
