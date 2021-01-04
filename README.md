# SegMyO
Segment my object - A pipeline to extract segmented objects in images based on labels or bounding boxes

![SegMyO pipeline](https://github.com/RobinDelearde/SegMyO/blob/main/SegMyO.png?raw=true)

The SegMyO (Segment my object) pipeline can be used to automatically extract segmented objects in images based on given labels and / or bounding boxes. When providing the expected label, the system looks for the closest label in the list of outputs, using a measure of semantic similarity. And when providing the boundingbox, it looks for the output object with the best coverage, based on several geometric criteria. Associated witha semantic segmentation model trained on a similar dataset, or a good region proposal algorithm, this pipeline provides a simple solution to segment efficiently a dataset without requiring specific training, but also to the problem of weakly-supervised segmentation. This is particularly useful to segment public datasets available with weak object annotations (e.g., bounding boxes and labels from a detection, labels from a caption) coming from an algorithm or from manual annotation.

This pipeline is made of the 2 parts:
- SegMyO_segmentation.py: segmentation with a pre-trained model
- SegMyO_selection.py: selection of the best output in a list of segment proposals, for a given bounding box and/or a given label, by using a combination of several geometric criteria on the bounding box and a semantic criterion if the expected label is given.

A test on a single image is given in SegMyO_test.py.

The code also provides a complete evaluation on PASCAL VOC 2012:
- PASCAL-VOC_segmentation_SegMyO.py: segmentation with SegMyO
- PASCAL-VOC_segmentation_full.py: segmentation with SegMyO + 2 baselines (GrabCut and fill_bbox)

If you use or adapt this code, thanks to cite this paper:
```
@inproceedings{delearde_visapp2021,
  author = {Deléarde, R. and Kurtz, C. and Dejean, P. and Wendling, L.},
  title = {Segment my object: A pipeline to extract segmented objects in images based on labels or bounding boxes},
  booktitle = {Int. Conf. on Computer Vision Theory and Applications (VISAPP)},
  year = {2021},
  pages = {XX--XX}
}
```
