# SegMyO
Segment My Object - A pipeline to extract segmented objects in images based on labels or bounding boxes

This pipeline is made of the 2 parts:
- SegMyO_segmentation.py: segmentation with a pre-trained model
- SegMyO_selection.py: selection of the best output in a list of segment proposals, for a given bounding box and/or a given label, by using a combination of several geometric criteria on the bounding box and a semantic criterion if the expected label is given.

A test on a single image is given in SegMyO_test.py.

The code also provides a complete evaluation on PASCAL VOC 2012:
- PASCAL-VOC_segmentation_SegMyO.py: segmentation with SegMyO
- PASCAL-VOC_segmentation_full.py: segmentation with SegMyO + 2 baselines (GrabCut and fill_bbox)

If you use or adapt this code, thanks to cite this paper:\
@inproceedings{delearde_visapp2021,\
author = {Deléarde, R. and Kurtz, C. and Dejean, P. and Wendling, L.},\
title = {Segment My Object - A pipeline to extract segmented objects in images based on labels or bounding boxes},\
booktitle = {Int. Conf. on Computer Vision Theory and Applications (VISAPP)},\
year = {2021},\
pages = {XX--XX}\
}
