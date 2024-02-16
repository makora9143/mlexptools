from typing import TypeVar, Iterable
import torch

from torcheval.metrics import Metric

TECE = TypeVar("TECE")


class MulticlassCalibrationError(Metric[torch.Tensor]):
    """
    Calculate the multiclass calibration error of a classifier.
    """

    def __init__(self: TECE, *, n_bins: int = 15, device=None) -> None:
        super().__init__(device=device)
        self.n_bins = n_bins
        self._add_state("confidences", torch.tensor([], device=self.device))
        self._add_state("accuracies", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self: TECE, preds: torch.Tensor, targets: torch.Tensor) -> TECE:
        if not torch.all((preds >= 0) * (preds <= 1)):
            preds = preds.softmax(1)
        confidences, predictions = preds.max(1)
        accuracies = predictions.eq(targets)
        self.confidences = torch.cat([self.confidences, confidences.float()])
        self.accuracies = torch.cat([self.accuracies, accuracies.float()])
        return self

    @torch.inference_mode()
    def compute(self: TECE) -> torch.Tensor:
        confidences = self.confidences
        accuracies = self.accuracies

        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, dtype=confidences.dtype, device=self.device)

        acc_bin = torch.zeros(len(bin_boundaries), dtype=confidences.dtype, device=self.device)
        conf_bin = torch.zeros(len(bin_boundaries), dtype=confidences.dtype, device=self.device)
        count_bin = torch.zeros(len(bin_boundaries), dtype=confidences.dtype, device=self.device)

        if self.device == torch.device('mps'):
            confidences = confidences.to('cpu')
            bin_boundaries = bin_boundaries.to('cpu')
            indices = torch.bucketize(confidences, bin_boundaries, right=True) - 1
            confidences = confidences.to(self.device)
            bin_boundaries = bin_boundaries.to(self.device)
            indices = indices.to(self.device)
        else:
            indices = torch.bucketize(confidences, bin_boundaries, right=True) - 1


        count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))

        conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
        conf_bin = torch.nan_to_num(conf_bin / count_bin)

        acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
        acc_bin = torch.nan_to_num(acc_bin / count_bin)

        prop_bin = count_bin / count_bin.sum()

        return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)

    @torch.inference_mode()
    def merge_state(self: TECE, metrics: Iterable[TECE]) -> TECE:
        confidences = [self.confidences, ]
        accuracies = [self.accuracies, ]

        for metric in metrics:
            confidences.append(metric.confidences)
            accuracies.append(metric.accuracies)
        self.confidences = torch.cat(confidences)
        self.accuracies = torch.cat(accuracies)
        return self