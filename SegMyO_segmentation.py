#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic segmentation of a dataset with several possible models,
based on segmentation models from:
    - https://pytorch.org/docs/stable/torchvision/models.html

author: Robin Deléarde @LIPADE (Université de Paris)
date: 22/12/2020
"""
#%% import libraries

from torchvision import models
from torchvision import transforms as T
# from torch import from_numpy

# preprocessing for FCN and DeepLabV3 torchvision models
preprocess_image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#%% semgentation function

# def seg_images(img, seg_model='Mask R-CNN'): # for a single image
def seg_images(tensor_images_list, seg_model='Mask R-CNN'):
    """
    Perform the semantic segmentation of a dataset, with several possible pre-trained models:
    (from https://pytorch.org/docs/stable/torchvision/models.html)
        - Mask R-CNN trained on COCO train2017
        - Faster R-CNN trained on COCO train2017
        - FCN trained on the subset of COCO train2017 which contains the same classes as Pascal VOC
        - DeepLabV3 trained on the subset of COCO train2017 which contains the same classes as Pascal VOC
    
    args:
    # array_images_list: array of shape [nb_images, nb_lines, nb_cols, nb_channels] containing images in range [0,255]
    tensor_images_list: tensor of shape [nb_images, nb_channels, nb_lines, nb_cols] containing images in range [0,1]
    seg_model: name of segmentation model, among ['Mask R-CNN' (default), 'Faster R-CNN', 'FCN', 'DeepLabV3']
    
    output:
    segmentation outputs dictionary structure with attributes:
    (from https://pytorch.org/docs/stable/torchvision/models.html)
        'boxes' (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format,
                with values of x between 0 and W and values of y between 0 and H
        'labels' (Int64Tensor[N]): the predicted labels for each image
        'scores' (Tensor[N]): the scores or each prediction
        'masks' (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range.
                In order to obtain the final segmentation masks, the soft masks can be thresholded,
                generally with a value of 0.5 (mask >= 0.5))
    """
    if seg_model in ['Mask R-CNN', 'Faster R-CNN', 'FCN', 'DeepLabV3']: # use Pytorch implementation and pre-trained models
        # tensor_images_list = [T.functional.to_tensor(img)] # shape [3, nb_lines, nb_cols], normalize to [0,1]
        # tensor_images_list = [from_numpy(img.transpose((2, 0, 1))/255.] # shape [3, nb_lines, nb_cols], normalize to [0,1]
        if seg_model == 'Mask R-CNN':
            model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval() # detection (bounding boxes) + segmentation
        elif seg_model == 'Faster R-CNN':
            models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval() # detection (bounding boxes) + segmentation
        elif seg_model == 'FCN':
            tensor_images_list = preprocess_image(tensor_images_list)
            model = models.segmentation.fcn_resnet101(pretrained=True).eval() # segmentation only
            # model = models.segmentation.fcn_resnet50(pretrained=True).eval() # segmentation only, "not supported as of now"
        elif seg_model == 'DeepLabV3':
            tensor_images_list = preprocess_image(tensor_images_list)
            model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval() # segmentation only
            # model = models.segmentation.deeplabv3_resnet50(pretrained=True).eval() # segmentation only, "not supported as of now"
        seg_output = model(tensor_images_list)[0]
    else:
        raise Exception('segmentation model not implemented')
    return seg_output
