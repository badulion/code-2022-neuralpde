from typing import Any, List

import torch
from torchdiffeq import odeint_adjoint, odeint
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from src.utils.metrics import RMSE

from src.models.components.simple_resnet import SimpleResnet
from src.models.components.nn_wrapper import NeuralNetWrapper
from src.models.components.hiddenstate import HiddenState

class HiddenStateModule(LightningModule):
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
        dynamic: torch.nn.Module,
        net: HiddenState,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        use_adjoint = False,
        solver = 'euler',
        step_size = '1/3'):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = NeuralNetWrapper(dynamic)
        self.projection = net
        self.solver = odeint_adjoint if use_adjoint else odeint

        self.input_dim = self.projection.input_dim
        self.hidden_dim = self.projection.hidden_dim

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
        x = x[-1,...]
        t_0 = torch.tensor([0], device=tpoints.device)
        t = torch.hstack([t_0, tpoints])
        x = self.projection(x)
        pred = self.solver(self.net, x, t, method=self.hparams.solver, options=dict(step_size=self.hparams.step_size))
        pred = pred[:,:,:self.input_dim]
        return pred[:-1]

    def neural_net_wrapper(self, t, x):
        return self.net(x)

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