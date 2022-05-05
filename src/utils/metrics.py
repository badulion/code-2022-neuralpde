import torchmetrics
import torch
from typing import Optional, Dict, Any


class RMSE(torchmetrics.Metric):
    def __init__(self, compute_on_step: Optional[bool] = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__(compute_on_step, **kwargs)
        # dist_reduce_func indiates the function that should be used to 
        # reduce state from multiple processes
        self.add_state("sum_squared_errors", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("n_observations", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        # update metric states
        self.sum_squared_errors += torch.sum((preds-target) ** 2)
        self.n_observations += preds.numel()

    def compute(self):
        # compute final result
        return torch.sqrt(self.sum_squared_errors / self.n_observations)