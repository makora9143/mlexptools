import sys
import json
import pathlib
import platform
import subprocess

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

from .utils import get_logger


_logger = get_logger(__name__)


class MPSMonitor(BaseMetricsMonitor):
    """ Class for monitoring MPS stats.

    This class is mirrored from wandb's mps (Apple GPU) module.
    https://github.com/wandb/wandb/blob/4b5a3be2dc028d84653c28d612cd96472278a5eb/wandb/sdk/internal/system/assets/gpu_apple.py#L40
    """
    MAX_POWER_WATTS = 16.5

    def __init__(self):

        super().__init__()

        self.binary_path = (
            pathlib.Path(sys.modules["mlexptools"].__path__[0]) / "bin" / "apple_gpu_stats"
        ).resolve()

    def collect_metrics(self):
        try:
            command = [str(self.binary_path), "--json"]
            output = (subprocess.check_output(command, universal_newlines=True).strip().split("\n"))[0]
            raw_stats = json.loads(output)

            self._metrics["mps_usage"].append(raw_stats["utilization"])
            self._metrics["mps_memory_allocated"].append(raw_stats["mem_used"])
            self._metrics["mps_temperature"].append(raw_stats["temperature"])
            self._metrics["mps_power_uatts"].append( raw_stats["power"])
            self._metrics["mps_power_percent"].append((raw_stats["power"] / self.MAX_POWER_WATTS) * 100)
        except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as e:
            _logger.error(f"Failed to collect MPS stats: {e}")

    def aggregate_metrics(self):
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}


class ExtendedSystemMetricsMonitor(SystemMetricsMonitor):
    """ Extended system metrics monitor to include MPS stats. """

    def __init__(self, run_id, sampling_interval=10, samples_before_logging=1, resume_logging=False):
        super().__init__(run_id, sampling_interval,samples_before_logging,resume_logging)

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            self.monitors.append(MPSMonitor())
