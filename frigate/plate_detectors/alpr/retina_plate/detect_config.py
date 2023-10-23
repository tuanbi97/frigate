import os

cfg_plate = {
    "path": os.path.dirname(os.path.realpath(__file__))
    + "/weight/Retina_Plate_dynamix_size.onnx",
    "path_pth": os.path.dirname(os.path.realpath(__file__))
    + "/weight/LP_detect_92.pth",
    "name": "mobilenet0.25",
    "dense_anchor": False,
    "min_dim": 512,
    "steps": [8, 16, 32],
    "min_sizes": [[10, 20], [32, 64], [128, 256]],
    "anchors": [
        16,
        20.2,
        25.4,
        32,
        40.3,
        50.8,
        64,
        80.6,
        101.6,
        128,
        161.3,
        203.2,
        256,
        322.54,
        406.37,
    ],
    "aspect_ratios": [[0.2, 0.5], [0.2, 0.5], [0.2, 0.5]],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": False,
    "gpu_inference": True,
    "batch_size": 4,
    "ngpu": 4,
    "epoch": 100,
    "decay1": 70,
    "decay2": 90,
    "image_size": (
        454,
        256,
    ),  # (350, 200),  # width, height  //(1980, 1020),  # width, height
    "max_size": 320,
    "image_ratio": [1, 1],
    "pretrain": False,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
    "in_channel": 32,
    "out_channel": 64,
    "confidence_threshold": 0.9,
    "nms_threshold": 0.4,
    "top_k": 5000,
    "keep_top_k": 750,
    "vis_thres": 0.4,
}
