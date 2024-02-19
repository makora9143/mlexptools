from typing import Optional, TypeVar

from torcheval.metrics import Mean
import torch

from ..utils import get_logger

TMean = TypeVar("TMean")
_logger = get_logger(__name__)


class MeanMetric(Mean):
    """
    Calculate the weighted mean value of all elements in all the input tensors.
    When weight is not provided, it calculates the unweighted mean.
    Its functional version is ``torcheval.functional.mean()``.
    Support for ``torch.device('mps')``.
    """

    def __init__(
        self: TMean,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._device = device
        # weighted sum of values over the entire state
        self._add_state(
            "weighted_sum", torch.tensor(0.0, device=self.device, dtype=torch.float32 if self.device == torch.device('mps') else torch.float64)
        )
        # sum total of weights over the entire state
        self._add_state(
            "weights", torch.tensor(0.0, device=self.device, dtype=torch.float32 if self.device == torch.device('mps') else torch.float64)
        )
