import torch
from typing import TypeVar

from torcheval.metrics import MulticlassAccuracy


TAgreement = TypeVar("TAgreement")


class MulticlassAgreement(MulticlassAccuracy):
    @torch.inference_mode()
    def update(self: TAgreement, input: torch.Tensor, target: torch.Tensor) -> TAgreement:
        target = target.argmax(1)
        return super().update(input, target)
