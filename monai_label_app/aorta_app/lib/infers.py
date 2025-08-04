import logging
import os
from collections.abc import Sequence
from typing import Callable, Optional

import torch
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
# BasicInferTaskをインポートします
from monailabel.tasks.infer.basic_infer import BasicInferTask # <--- 変更
from monailabel.utils.others.generic import gpu_memory_map

from .configs import INFERENCE_CONFIG, INTENSITY_RANGE, NETWORK_CONFIG, TARGET_SPACING

logger = logging.getLogger(__name__)


# 継承クラスを BasicInferTask に変更します
class AortaSegmentation(BasicInferTask): # <--- 変更
    """
    Inference task for aortic segmentation using 3D UNet
    """

    def __init__(
        self,
        path: str,
        network: Optional[torch.nn.Module] = None,
        roi_size: Sequence[int] = (96, 96, 96),
        preload: bool = False,
        config: Optional[dict] = None,
    ):
        # super().__init__ に必須引数を追加します
        super().__init__( # <--- 変更
            path=path,
            network=network,
            type="segmentation", # <--- 追加
            labels={"background": 0, # キーと値を入れ替え、値は整数に
                    "label_1": 1,
                    "label_2": 2,
                    "label_3": 3,
                    "label_4": 4,
                    "label_5": 5,
                    "label_6": 6,
                    "label_7": 7,
                    "label_8": 8,
                    "label_9": 9,
                    "label_10": 10,
                    "label_11": 11,
                    "label_12": 12,
                    "label_13": 13,
                    "label_14": 14,
                    "label_15": 15,
                    "label_16": 16,
                    "label_17": 17,
                    "label_18": 18,
                    "label_19": 19,
                    "label_20": 20,
                    "label_21": 21,
                    "label_22": 22,
                    "label_23": 23}, # 変更
            dimension=3, # <--- 追加
            description="A 3D UNet model for aortic segmentation", # <--- 追加
            roi_size=roi_size,
            preload=preload,
            config=config,
        )

        self.roi_size = roi_size

        if network is None:
            self.network = UNet(**NETWORK_CONFIG)
        else:
            self.network = network
            
        
    def pre_transforms(self, data: Optional[dict] = None) -> list:  # <--- listを返すように変更
        """
        Pre-processing transforms matching the training pipeline
        """
        # Composeでラップせず、リストを直接返します
        return [ # <--- 変更
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityRanged(
                keys="image",
                a_min=INTENSITY_RANGE["a_min"],
                a_max=INTENSITY_RANGE["a_max"],
                b_min=INTENSITY_RANGE["b_min"],
                b_max=INTENSITY_RANGE["b_max"],
                clip=True,
            ),
            CropForegroundd(keys="image", source_key="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(
                keys="image",
                pixdim=TARGET_SPACING,
                mode="bilinear",
            ),
            EnsureTyped(keys="image"),
        ] # <--- 変更

    def inferer(self, data: Optional[dict] = None) -> Callable:
        """
        Sliding window inferer for large volume inference
        """
        return SlidingWindowInferer(
            roi_size=INFERENCE_CONFIG["roi_size"],
            sw_batch_size=INFERENCE_CONFIG["sw_batch_size"],
            overlap=INFERENCE_CONFIG["overlap"],
        )

    def inverse_transforms(self, data: Optional[dict] = None) -> list: # <--- listを返すように変更
        """
        Inverse transforms to restore original spacing and orientation
        """
        # Composeでラップせず、リストを直接返します
        return [] 

    def post_transforms(self, data: Optional[dict] = None) -> list: # <--- listを返すように変更
        """
        Post-processing transforms
        """
        # Composeでラップせず、リストを直接返します
        return [ # <--- 変更
            EnsureTyped(keys="pred", device="cuda", track_meta=False),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            EnsureTyped(keys="pred", device="cpu"),
        ] 
    
    def __call__(self, request: dict, datastore: Optional[dict] = None) -> dict:
        """
        Execute inference on the input request
        """


        logger.info("Starting aortic segmentation inference")

        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {gpu_memory_map()}")

        result = super().__call__(request, datastore)

        logger.info("Aortic segmentation inference completed")
        return result

    def get_path(self) -> str:
        """
        Get the model path
        """
        return self.path

    def is_valid(self) -> bool:
        """
        Check if the model file exists and is valid
        """
        paths = self.path if isinstance(self.path, list) else [self.path]
        for p in paths:
            # pathがNoneまたは空文字列の場合も考慮
            if not p or not os.path.exists(p):
                logger.warning(f"Model file not found or path is invalid: {p}")
                return False
        return True

    def get_config(self) -> dict:
        """
        Get inference configuration
        """
        return {
            "network": NETWORK_CONFIG,
            "inference": INFERENCE_CONFIG,
            "intensity_range": INTENSITY_RANGE,
            "target_spacing": TARGET_SPACING,
            "roi_size": self.roi_size,
        }
    
    def _get_network(self, device, data):
        """
        Overrides the parent method to load a TorchScript model correctly.
        """
        # self.pathがリストの場合は最初の要素を、文字列の場合はそのまま使用
        path = self.path[0] if isinstance(self.path, list) and self.path else self.path
        if not path:
            return super()._get_network(device, data)

        # TorchScriptモデルをロードするための推奨される方法である torch.jit.load を使用します
        # これにより、weights_only の問題を回避できます
        logger.info(f"Loading TorchScript model from: {path}")
        network = torch.jit.load(path, map_location=torch.device(device))
        
        # ネットワークは既にロードされているので、それを返すだけです
        return network
