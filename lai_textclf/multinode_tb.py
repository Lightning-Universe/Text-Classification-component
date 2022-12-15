from typing import Type

import lightning as L

from lai_textclf.tensorboard import TensorBoardWork


class MultiNodeLightningTrainerWithTensorboard(L.LightningFlow):
    def __init__(
        self,
        work_cls: Type[L.LightningWork],
        num_nodes: int,
        cloud_compute: L.CloudCompute,
    ):
        super().__init__()
        tb_drive = L.app.storage.Drive("lit://tb_drive")
        self.tensorboard_work = TensorBoardWork(drive=tb_drive)
        self.text_classificaion = L.app.components.LightningTrainerMultiNode(
            work_cls,
            num_nodes=num_nodes,
            cloud_compute=cloud_compute,
            tb_drive=tb_drive,
        )

    def run(self, *args, **kwargs) -> None:
        self.tensorboard_work.run()
        self.text_classificaion.run()

    def configure_layout(self):
        return [{"name": "Training Logs", "content": self.tensorboard_work.url}]
