# Script to run inference on small test WSIs
# Can easily adjusted by changing
# 1) the data_dir your directory containing WSIs
# 2) the glob_pattern to find the WSIs
# 3) Changing the mpp to 2.0 (which the model is trained on) and to get manageable output.

# =====         Step 1       =====
# If on CPU, set `trainer.accelerator=cpu` and set map_location to cpu in utils.io.load_weights
# If on MacOS, set multiprocessing.set_start_method("fork", force=True) before main() in inference.py. mps is not supported.

# =====         Step 2       =====
export SCRATCH=/set/to/your/log/dir

# =====         Step 3       =====
# Set path to model checkpoint
# Expected to be an attention unet with current configuration.
# ABSOLUTE_PATH_TO_CKPT=/absolute/path/to/checkpoint.ckpt

# =====         Step 4       =====
# Create ~/ahcore/additional_config/data_description/on_the_fly_data_description.yaml
# With the following contents

#~/ahcore/additional_config/data_description/on_the_fly_data_description.yaml
# --------------------------------------------------------------------------------- #
# _target_: ahcore.utils.data.OnTheFlyDataDescription #TODO check if this works
# data_dir: test_in_memory_db # Root dir for images to be found
# glob_pattern: "**/*.svs" # Glob pattern relative to root dir to find images
# use_roi: False
# apply_color_profile: False
# inference_grid:
#   mpp: 0.01 # Since the images are VERY small, we need to blow them up to be able to get proper input image size
#   tile_size: [512, 512]
#   tile_overlap: [0, 0]

# num_classes: 2

# remap_labels:
#   specimen: specimen

# index_map:
#   specimen: 1

# color_map:
#   0: "yellow"
#   1: "red"
# --------------------------------------------------------------------------------- #


# =====         Step 5       =====
# Run this file from ~/ahcore/tools as CWD, with an installed ahcore environment.
python inference.py \
    callbacks=inference \
    data_description=on_the_fly_data_description \
    pre_transform=segmentation \
    augmentations=segmentation \
    lit_module=monai_segmentation/attention_unet \
    ckpt_path=$ABSOLUTE_PATH_TO_CKPT \
    # trainer.accelerator=cpu # If you're running on CPU

# =====         Step 6       =====
# Files should be saved in $SCRATCH / segmentation_inference / $datetime / outputs / AttentionUNet / 0_0 / *
