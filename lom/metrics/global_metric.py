from typing import List
import torch
from torch import Tensor
from torchmetrics import Metric

class GlobalMetrics(Metric):
    def __init__(self,
                 cfg,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.name = "Global metrics"

        # Add states for accumulating metrics
        self.add_state("MRPE_x", default=torch.tensor(0.0), dist_reduce_fx="sum")     
        self.add_state("MRPE_y", default=torch.tensor(0.0), dist_reduce_fx="sum")     
        self.add_state("MRPE_z", default=torch.tensor(0.0), dist_reduce_fx="sum")     
        self.add_state("MRPE_all", default=torch.tensor(0.0), dist_reduce_fx="sum")     
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def calc_MRPE(self, preds, target):
        """
        Mean per-joint position error (i.e. mean Euclidean distance)
        often referred to as "Protocol #1" in many papers.
        """
        return torch.norm(preds - target, p=2, dim=-1)

    @torch.no_grad()
    def update(self, rec_trans: Tensor, tar_trans: Tensor, lengths: Tensor = None):

        bs, n = rec_trans.shape[0], rec_trans.shape[1]
        self.count += bs
        # Calculate metrics
        for bs_idx in range(bs):
            length = lengths[bs_idx]
            self.MRPE_x += self.calc_MRPE(rec_trans[bs_idx, :length, 0:1] * 1000, tar_trans[bs_idx, :length, 0:1] * 1000).mean()
            self.MRPE_y += self.calc_MRPE(rec_trans[bs_idx, :length, 1:2] * 1000, tar_trans[bs_idx, :length, 1:2] * 1000).mean()
            self.MRPE_z += self.calc_MRPE(rec_trans[bs_idx, :length, 2:3] * 1000, tar_trans[bs_idx, :length, 2:3] * 1000).mean()
            self.MRPE_all += self.calc_MRPE(rec_trans[bs_idx, :length] * 1000, tar_trans[bs_idx, :length] * 1000).mean()

    def compute(self, sanity_flag=False):
        """Compute final metrics.
        Args:
            sanity_flag: Flag for sanity check, ignored in computation
        """
        metrics = {
            "MRPE_x": self.MRPE_x / self.count,
            "MRPE_y": self.MRPE_y / self.count,
            "MRPE_z": self.MRPE_z / self.count,
            "MRPE_all": self.MRPE_all / self.count
        }
        return metrics 