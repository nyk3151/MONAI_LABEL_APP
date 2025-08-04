PRETRAINED_PATH = {
    "aorta_segmentation_unet": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/aorta_segmentation_unet_v1.0.0.zip"
}

SPATIAL_SIZE = [96, 96, 96]

INTENSITY_RANGE = {
    "a_min": -175,  # Minimum HU value
    "a_max": 250,  # Maximum HU value
    "b_min": 0.0,  # Scaled minimum
    "b_max": 1.0,  # Scaled maximum
}

TARGET_SPACING = [1.5, 1.5, 2.0]  # mm

NETWORK_CONFIG = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 24,  # Background + 23 anatomical regions
    "channels": [16, 32, 64, 128, 256],
    "strides": [2, 2, 2, 2],
    "num_res_units": 2,
}

INFERENCE_CONFIG = {"roi_size": SPATIAL_SIZE, "sw_batch_size": 1, "overlap": 0.5}

LABEL_NAMES = {
    0: "background",
    1: "aortic_root",
    2: "ascending_aorta",
    3: "aortic_arch",
    4: "descending_aorta",
    5: "abdominal_aorta",
    6: "brachiocephalic_trunk",
    7: "left_common_carotid",
    8: "left_subclavian",
    9: "right_common_carotid",
    10: "right_subclavian",
    11: "celiac_trunk",
    12: "superior_mesenteric",
    13: "left_renal",
    14: "right_renal",
    15: "inferior_mesenteric",
    16: "left_common_iliac",
    17: "right_common_iliac",
    18: "left_internal_iliac",
    19: "right_internal_iliac",
    20: "left_external_iliac",
    21: "right_external_iliac",
    22: "region_22",
    23: "region_23",
}
