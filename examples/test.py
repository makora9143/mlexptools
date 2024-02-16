import time
import torch
from torcheval import metrics as tm

from mlexptools.writer import MlflowWriter
import mlexptools


class EpochMetrics(mlexptools.metrics.AbstractMetricCollection):
    acc: tm.MulticlassAccuracy = tm.MulticlassAccuracy()
    loss: tm.Mean = tm.Mean()


metrics = EpochMetrics()
writer = MlflowWriter(exp_name="test", run_name="hoge")
writer.log_params(dict(
    lr=0.01,
    batch_size=256,
    epochs=100,
))

for epoch in range(3):
    for i in range(11):
        time.sleep(1)
        metrics.update(
            dict(
                acc=(torch.randn(3, 5), torch.randint(0, 5, (3,))),
                loss=torch.rand(1)
            )
        )
    writer.log_metrics(
        metrics.compute(),
        step=epoch+1
    )
    metrics.reset()

writer.finish()