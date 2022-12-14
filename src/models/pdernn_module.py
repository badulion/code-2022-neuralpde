from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from src.utils.metrics import RMSE

from src.models.components.pdernn import PDERNN


class PDERNNModule(LightningModule):
    """Example of LightningModule for Learning PDEs.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: PDERNN,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        use_last_only: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.use_last_only = use_last_only

        # loss function
        self.criterion = torch.nn.MSELoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metric = RMSE()
        self.val_metric = RMSE()
        self.test_metric = RMSE()

        # for logging best so far validation accuracy
        self.val_metric_best = MinMetric()

    def forward(self, x: torch.Tensor, tpoints: torch.Tensor):
        # in lightning, forward defines the prediction/inference actions
        if self.use_last_only:
            x = x[-1,...]
            x = x.unsqueeze(0).expand(len(tpoints), -1, -1, -1, -1)
        x = torch.transpose(x, 0, 1)
        pred = self.net(x)
        pred = torch.transpose(pred, 0, 1)
        return pred

    def step(self, batch: Any):
        x, y, tpoints = batch
        preds = self.forward(x, tpoints)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        rmse = self.train_metric(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/rmse", rmse, on_step=True, on_epoch=False, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss} # {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        rmse = self.val_metric(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss} # {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        rmse = self.val_metric.compute()  # get val accuracy from current epoch
        self.val_metric_best.update(rmse)
        self.log("val/rmse_best", self.val_metric_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        rmse = self.test_metric(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/rmse", rmse, on_step=False, on_epoch=True)

        mse = torch.mean((preds-targets)**2, dim=(-1, -2))

        return {"loss": loss, "mse": mse}

    def test_epoch_end(self, outputs: List[Any]):
        # concatenate outputs
        mse = torch.concat([d['mse'] for d in outputs], dim=1)
        horizon_mse = torch.mean(mse, dim=1)
        
        # log horizon metrics
        horizon = horizon_mse.size(dim=0)
        target_names = list(self.trainer.datamodule.target_dict.keys()) # hack
        for i in range(horizon):
            metrics = {f"test/rmse/{target_names[k]}": torch.sqrt(horizon_mse[i, k]) for k in range(len(target_names))}
            for logger in self.loggers:
                logger.log_metrics(metrics, step=i+1)

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_metric.reset()
        self.test_metric.reset()
        self.val_metric.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
