from typing import List
import os
import torch
from torch import Tensor
from torchmetrics import Metric
from .utils import *
from bert_score import score as score_bert
import spacy
from lom.config import instantiate_from_config
import jiwer

class A2TMetrics(Metric):
    def __init__(self,
                 cfg,
                 dataname='librispeech',
                 dist_sync_on_step=True, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.dataname = dataname
        # Initialize states for WER and CER
        self.add_state("wer_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("cer_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("avg_wer", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("avg_cer", default=torch.tensor(0.0), dist_reduce_fx="sum")



    @torch.no_grad()
    def update(self, pred_texts: List[str], gt_texts: List[str]):
        assert len(pred_texts) == len(gt_texts), "The length of predictions and ground truths must match."

        batch_size = len(pred_texts)

        wer_batch = torch.tensor(0.0)
        cer_batch = torch.tensor(0.0)

        for pred, gt in zip(pred_texts, gt_texts):
            wer = jiwer.wer(gt, pred)
            cer = jiwer.cer(gt, pred)

            wer_batch += wer
            cer_batch += cer

        self.wer_sum += wer_batch
        self.cer_sum += cer_batch
        self.count += batch_size

        self.avg_wer = self.wer_sum / self.count
        self.avg_cer = self.cer_sum / self.count

    @torch.no_grad()
    def compute(self, sanity_flag=False):
        return {"avg_wer": self.avg_wer, "avg_cer": self.avg_cer}

    @torch.no_grad()
    def reset(self):
        self.wer_sum = torch.tensor(0.0)
        self.cer_sum = torch.tensor(0.0)
        self.count = torch.tensor(0)
        self.avg_wer = torch.tensor(0.0)
        self.avg_cer = torch.tensor(0.0)