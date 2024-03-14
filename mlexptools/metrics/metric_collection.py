from typing import Optional, Union, Dict, Dict
from copy import deepcopy
from abc import ABC
from dataclasses import dataclass, asdict

from torch import Tensor
from torcheval import metrics as tm


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


@dataclass
class AbstractMetricCollection(ABC):
    """_summary_
    Your metrics should subclass this class.

    class MultiCollection(AbstractMetricCollection):
        training_acc: tm.MulticlassAccuracy
        training_loss: tm.Mean

    """

    def __new__(cls, *args, **kwargs):
        dataclass(cls, repr=False)
        return super().__new__(cls)

    def __repr__(self):
        main_str = f"{self.__class__.__name__}("
        child_lines = []
        for key, module in self.__dict__.items():
            mod_str = module.__class__.__name__
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)

        main_str += '\n  ' + '\n  '.join(child_lines) + '\n'
        main_str += ')'

        return main_str

    def __post_init__(self) -> None:
        self._type_check()

    def _type_check(self) -> None:
        _dict = asdict(self)

        for field, field_type in self.__annotations__.items():
            if not isinstance(_dict[field], tm.Metric):
                raise TypeError(f"Field type '{field_type}' must be child class of torcheval.metrics.Metric, and the value of {field} be {field_type} not {_dict[field]}.")

    @property
    def metrics(self):
        return vars(self)

    def update(self, metrics: Dict[str, Union[Tensor, tuple[Tensor]]]) -> None:
        for k, v in metrics.items():
            if k not in self.metrics:
                raise KeyError(f"Unknown metric key {k}")

            if isinstance(v, (list, tuple)):
                self.metrics[k].update(*v)
            else:
                self.metrics[k].update(v)

    def compute(self) -> Dict[str, float]:
        output = {}
        for k, v in self.metrics.items():
            if isinstance(v, tm.Cat):
                continue
            value = v.compute().item()
            output[k] = value
        return output

    def reset(self):
        for v in self.metrics.values():
            v.reset()

    def compute_with_prefix(self, prefix: str):
        output = self.compute()
        if prefix is not None:
            output = {f'{prefix}_{k}': v for k, v in output.items()}
        return output

    def clone(self) -> "AbstractMetricCollection":
        mh = deepcopy(self)
        return mh


@dataclass
class MetricCollection:
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.__dict__.update(kwargs)

    def __new__(cls, *args, **kwargs) -> "MetricCollection":
        dataclass(cls, repr=False)
        return super().__new__(cls)

    def __repr__(self) -> str:
        main_str = f"{self.__class__.__name__}("
        child_lines = []
        for key, module in self.__dict__.items():
            mod_str = module.__class__.__name__
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)

        main_str += '\n  ' + '\n  '.join(child_lines) + '\n'
        main_str += ')'

        return main_str

    @property
    def metrics(self) -> Dict[str, tm.Metric]:
        return vars(self)

    def update(self, metrics: Dict[str, Union[Tensor, tuple[Tensor]]]) -> None:
        for k, v in metrics.items():
            if k not in self.metrics:
                raise KeyError(f"Unknown metric key {k}")

            if isinstance(v, (list, tuple)):
                self.metrics[k].update(*v)
            else:
                self.metrics[k].update(v)

    def compute(self) -> Dict[str, float]:
        output = {}
        for k, v in self.metrics.items():
            if isinstance(v, tm.Cat):
                continue
            value = v.compute().item()
            output[k] = value
        return output

    def reset(self):
        for v in self.metrics.values():
            v.reset()

    def compute_with_prefix(self, prefix: str) -> Dict[str, float]:
        output = self.compute()
        if prefix is not None:
            output = {f'{prefix}_{k}': v for k, v in output.items()}
        return output

    def clone(self) -> "MetricCollection":
        mh = deepcopy(self)
        return mh
