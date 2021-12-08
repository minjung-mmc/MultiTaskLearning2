import torch


class params:
    def __init__(self):
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.train_img_path = "../dataset/CityScape/leftImg8bit/train/"
        self.train_seg_label = "../dataset/CityScape/gtFine/train/"
        self.train_depth_label = "../dataset/CityScape/disparity/train/"
        self.val_img_path = "../dataset/CityScape/leftImg8bit/val/"
        self.val_seg_label = "../dataset/CityScape/gtFine/val/"
        self.val_depth_label = "../dataset/CityScape/disparity/val/"
        self.test_img_path = "../dataset/CityScape/leftImg8bit/test/"
        self.test_seg_label = "../dataset/CityScape/gtFine/test/"
        self.test_depth_label = "../dataset/CityScape/disparity/test/"
        self.val_ann = "../dataset/CityScape/customized/cityscapes_panoptic_val.json"
        self.train_ann = (
            "../dataset/CityScape/customized/cityscapes_panoptic_train.json"
        )
        self.batch_size = 1
        self.mode = "Train"
        self.num_epoch = 100
        self.gamma = 0.1
        self.lr = 0.0005
        self.num_classes_seg = 20  # 원래 34개
        self.num_classes_depth = 1
        self.beta1 = 0.5
        self.depth_weight = 8
        self.od_weight = 0.33
        self.d5 = {
            "name": "tf_efficientdet_d5",
            "backbone_name": "tf_efficientnet_b5",
            "backbone_args": {"drop_path_rate": 0.2},
            "backbone_indices": None,
            "image_size": [1024, 2048],
            "num_classes": 34,
            "min_level": 3,
            "max_level": 7,
            "num_levels": 5,
            "num_scales": 3,
            "aspect_ratios": [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]],
            "anchor_scale": 4.0,
            "pad_type": "same",
            "act_type": "swish",
            "norm_layer": None,
            "norm_kwargs": {"eps": 0.001, "momentum": 0.01},
            "box_class_repeats": 4,
            "fpn_cell_repeats": 7,
            "fpn_channels": 288,
            "separable_conv": True,
            "apply_resample_bn": True,
            "conv_after_downsample": False,
            "conv_bn_relu_pattern": False,
            "use_native_resize_op": False,
            "downsample_type": "max",
            "upsample_type": "nearest",
            "redundant_bias": True,
            "head_bn_level_first": False,
            "head_act_type": None,
            "fpn_name": None,
            "fpn_config": None,
            "fpn_drop_path_rate": 0.0,
            "alpha": 0.25,
            "gamma": 1.5,
            "label_smoothing": 0.0,
            "legacy_focal": False,
            "jit_loss": False,
            "delta": 0.1,
            "box_loss_weight": 50.0,
            "soft_nms": False,
            "max_detection_points": 5000,
            "max_det_per_image": 100,
            "url": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5_51-c79f9be6.pth",
            "load_network": False,
            "load_network_path": "./weights/2021-11-22 19:34",
        }

        self.d0 = {
            "name": "tf_efficientdet_d0",
            "backbone_name": "tf_efficientnet_b0",
            "backbone_args": {"drop_path_rate": 0.2},
            "backbone_indices": None,
            "image_size": [1024, 2048],  # about pred # h, w #
            "num_classes": 34,  #
            "min_level": 3,
            "max_level": 7,
            "num_levels": 5,
            "num_scales": 3,
            "aspect_ratios": [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]],
            "anchor_scale": 4.0,
            "pad_type": "same",
            "act_type": "swish",
            "norm_layer": None,
            "norm_kwargs": {"eps": 0.001, "momentum": 0.01},
            "box_class_repeats": 3,  #
            "fpn_cell_repeats": 3,  #
            "fpn_channels": 64,  #
            "separable_conv": True,
            "apply_resample_bn": True,
            "conv_after_downsample": False,
            "conv_bn_relu_pattern": False,
            "use_native_resize_op": False,
            "downsample_type": "max",
            "upsample_type": "nearest",
            "redundant_bias": True,
            "head_bn_level_first": False,
            "head_act_type": None,
            "fpn_name": None,
            "fpn_config": None,
            "fpn_drop_path_rate": 0.0,
            "alpha": 0.25,
            "gamma": 1.5,
            "label_smoothing": 0.0,
            "legacy_focal": False,
            "jit_loss": False,
            "delta": 0.1,
            "box_loss_weight": 50.0,
            "soft_nms": False,
            "max_detection_points": 5000,
            "max_det_per_image": 100,
            "url": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_34-f153e0cf.pth",
            "load_network": False,
            "load_network_path": "./weights/2021-12-06 21:53",
        }
        self.d4 = {
            "name": "tf_efficientdet_d4",
            "backbone_name": "tf_efficientnet_b4",
            "backbone_args": {"drop_path_rate": 0.2},
            "backbone_indices": None,
            "image_size": [1024, 2048],
            "num_classes": 34,
            "min_level": 3,
            "max_level": 7,
            "num_levels": 5,
            "num_scales": 3,
            "aspect_ratios": [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]],
            "anchor_scale": 4.0,
            "pad_type": "same",
            "act_type": "swish",
            "norm_layer": None,
            "norm_kwargs": {"eps": 0.001, "momentum": 0.01},
            "box_class_repeats": 4,
            "fpn_cell_repeats": 7,
            "fpn_channels": 224,
            "separable_conv": True,
            "apply_resample_bn": True,
            "conv_after_downsample": False,
            "conv_bn_relu_pattern": False,
            "use_native_resize_op": False,
            "downsample_type": "max",
            "upsample_type": "nearest",
            "redundant_bias": True,
            "head_bn_level_first": False,
            "head_act_type": None,
            "fpn_name": None,
            "fpn_config": None,
            "fpn_drop_path_rate": 0.0,
            "alpha": 0.25,
            "gamma": 1.5,
            "label_smoothing": 0.0,
            "legacy_focal": False,
            "jit_loss": False,
            "delta": 0.1,
            "box_loss_weight": 50.0,
            "soft_nms": False,
            "max_detection_points": 5000,
            "max_det_per_image": 100,
            "url": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d4_49-f56376d9.pth",
            "load_network": False,
            "load_network_path": "./weights/2021-11-22 19:34",
        }

        self.d3 = {
            "name": "tf_efficientdet_d3",
            "backbone_name": "tf_efficientnet_b3",
            "backbone_args": {"drop_path_rate": 0.2},
            "backbone_indices": None,
            "image_size": [1024, 2048],
            "num_classes": 34,
            "min_level": 3,
            "max_level": 7,
            "num_levels": 5,
            "num_scales": 3,
            "aspect_ratios": [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]],
            "anchor_scale": 4.0,
            "pad_type": "same",
            "act_type": "swish",
            "norm_layer": None,
            "norm_kwargs": {"eps": 0.001, "momentum": 0.01},
            "box_class_repeats": 4,
            "fpn_cell_repeats": 6,
            "fpn_channels": 160,
            "separable_conv": True,
            "apply_resample_bn": True,
            "conv_after_downsample": False,
            "conv_bn_relu_pattern": False,
            "use_native_resize_op": False,
            "downsample_type": "max",
            "upsample_type": "nearest",
            "redundant_bias": True,
            "head_bn_level_first": False,
            "head_act_type": None,
            "fpn_name": None,
            "fpn_config": None,
            "fpn_drop_path_rate": 0.0,
            "alpha": 0.25,
            "gamma": 1.5,
            "label_smoothing": 0.0,
            "legacy_focal": False,
            "jit_loss": False,
            "delta": 0.1,
            "box_loss_weight": 50.0,
            "soft_nms": False,
            "max_detection_points": 5000,
            "max_det_per_image": 100,
            "url": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3_47-0b525f35.pth",
            "load_network": False,
            "load_network_path": "./weights/2021-11-22 19:34",
        }


params = params()


if __name__ == "__main__":

    params = params()
    print(params.city)
