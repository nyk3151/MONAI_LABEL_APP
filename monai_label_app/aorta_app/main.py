import logging
import os

from lib.infers import AortaSegmentation
from monai.networks.nets import UNet
from monailabel.interfaces.app import MONAILabelApp
# TaskConfig は不要なので削除
# from monailabel.interfaces.config import TaskConfig 
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import strtobool

logger = logging.getLogger(__name__)


class AortaApp(MONAILabelApp):
    """
    MONAI Label App for Aortic Segmentation using 3D UNet
    """

    def __init__(self, app_dir: str, studies: str, conf: dict):
        # model_dirのパスを 'model' から 'models' に修正（一般的な命名規則に合わせる）
        self.model_dir = os.path.join(app_dir, "models")

        # ###############################################################
        # #### 以下の TaskConfig を使ったブロックはすべて削除します ####
        # ###############################################################
        # configs = {}
        # for t in ["segmentation"]:
        #     configs[t] = TaskConfig(...)
        #
        # models = { ... }
        # if conf.get("use_pretrained_model", True):
        #    ...

        # シンプルな super().__init__ 呼び出しに変更
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="Aortic Segmentation App",
            description="MONAI Label App for Aortic Segmentation using 3D UNet",
            version="1.0.0",
        )

    def init_infers(self) -> dict[str, InferTask]:
        """
        Initialize inference tasks
        """
        # infers: dict[str, InferTask] = {}

        # # 'aorta_segmentation' はサーバー起動時の --conf models の名前と一致させます
        # infers["aorta_segmentation"] = AortaSegmentation(
        #     # モデルパスを self.model_dir を使うように修正
        #     path=os.path.join(self.model_dir, "model.pt"), # best_metric_model.pth から変更
        #     network=self._get_network(),
        #     roi_size=self.conf.get("spatial_size", [96, 96, 96]),
        #     preload=strtobool(self.conf.get("preload", "false")),
        #     config={"cache_transforms": True, "cache_transforms_in_memory": True},
        # )

        # return infers

        return {
            "aorta_segmentation": AortaSegmentation(
                path=os.path.join(self.model_dir, "model.pt"),
            )
        }

    def init_trainers(self) -> dict[str, TrainTask]:
        """
        Initialize training tasks
        """
        return {} # 学習タスクがない場合は空の辞書を返す

    def _get_network(self) -> UNet:
        """
        Get the 3D UNet network definition
        """
        from monai.networks.nets import UNet

        return UNet(
            spatial_dims=3,
            in_channels=1,
            # ↓↓↓【注意】↓↓↓
            out_channels=24,  # 背景(0) + 24領域 = 25クラス
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def init_datastore(self) -> Datastore:
        """
        Initialize datastore
        """
        return super().init_datastore()