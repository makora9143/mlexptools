from typing import Union, Optional, TypeVar

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
        super().__init__(device=device)
        # weighted sum of values over the entire state
        self._add_state(
            "weighted_sum", torch.tensor(0.0, device=self.device, dtype=torch.float32 if self.device == torch.device('mps') else torch.float64)
        )
        # sum total of weights over the entire state
        self._add_state(
            "weights", torch.tensor(0.0, device=self.device, dtype=torch.float32 if self.device == torch.device('mps') else torch.float64)
        )

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMean,
        input: torch.Tensor,
        *,
        weight: Union[float, int, torch.Tensor] = 1.0,
    ) -> TMean:
        """
        Compute weighted mean. When weight is not provided, it calculates the unweighted mean.

        weighted_mean = sum(weight * input) / sum(weight)

        Args:
            input (Tensor): Tensor of input values.
            weight(optional): Float or Int or Tensor of input weights. It is default to 1.0. If weight is a Tensor, its size should match the input tensor size.
        Raises:
            ValueError: If value of weight is neither a ``float`` nor a ``int'' nor a ``torch.Tensor`` that matches the input tensor size.
        """
        if input.device != self.device:
            _logger.warning(
                f"input tensor is on {input.device} while the metric is on {self.device}."
                "Moving the metric to the device of the input tensor."
            )
            self.weighted_sum = self.weighted_sum.to(input)
            self.weights = self.weights.to(input)

        super().update(input, weight=weight)


