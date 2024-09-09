.. role:: raw-html-m2r(raw)
   :format: html


Model Zoo and Baselines
=======================

Introduction
------------

This file documents models associated with the ahcore project. You can download the parameters and weights of these
models, provided as a zip file.

License
-------

All models made available through this page are licensed under the\ :raw-html-m2r:`<br>`
`Creative Commons Attribution-ShareAlike 3.0 license <https://creativecommons.org/licenses/by-sa/3.0/>`_.

Baselines
---------

Segmentation models
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Model
     - Tissue type
     - Staining
     - Checkpoint
     - Version
   * - U-net
     - pancancer
     - H&E
     - `download <https://files.aiforoncology.nl/ahcore/v0.1.1/tissue_background.zip>`_
     - v0.1.1

Training tissue foreground segmentation model
=============================================

Introduction
------------
This file documents the training of a tissue foreground segmentation model. The model is trained on a dataset consisting of whole slide images (WSIs) of Camelyon16 and TCGA-BRCA datasets.

The following steps are necessary to train the model:
1. Creation of a manifest file containing the paths to the WSIs and their corresponding annotations.
2. Configuration of ahcore to train the foreground segmentation model.

Create a manifest file
----------------------
In order to create a manifest file to train a segmentation model, use the cli utility provided here `<https://github.com/NKI-AI/pathology-data-utilities/tree/main/ahcore/tissue_background>`_.

Configure ahcore for training the model
---------------------------------------
Use the following data_description to train the model:

```
_target_: ahcore.utils.data.DataDescription
data_dir: ${oc.env:DATA_DIR}
annotations_dir: ${oc.env:ANNOTATIONS_DIR} # specify in .env
manifest_database_uri: sqlite:////${oc.env:MANIFEST_PATH}/tcga_database.db
mask_label: null
mask_threshold: null
manifest_name: "v20231115_tcga_camelyon_bg"
split_version: "v1"
roi_name: null  # This is the name of the label that carries the ROI
use_roi: False
apply_color_profile: False
# Eg. 512 x 512 is: tile_size 726 x 726 and tile_overlap 107 x 107
# Tiles are cropped in the transforms downstream, to ensure the patch is completely visible, they are extracted
# slightly larger qrt 2) with sufficient overlap so we have a 512 stride.
training_grid:
  mpp: 8
  output_tile_size: [726, 726]
  tile_overlap: [0, 0]
  tile_size: [512, 512]
inference_grid:
  mpp: 8
  tile_size: [512, 512]
  tile_overlap: [128, 128]

num_classes: 2
use_class_weights: false  # Use the class weights in the loss

remap_labels:
  specimen: specimen

index_map:
  specimen: 1

color_map:
  0: "yellow"
  1: "red"
```
