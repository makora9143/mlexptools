from typing import Dict, Optional, Union
import numpy
import matplotlib
import PIL

from omegaconf import DictConfig

from mlflow.system_metrics import system_metrics_monitor
import mlflow

from .utils import get_logger, omegaconf_to_dict
from .mps_monitor import ExtendedSystemMetricsMonitor


system_metrics_monitor.SystemMetricsMonitor = ExtendedSystemMetricsMonitor
_logger = get_logger(__name__)


class MlflowWriter:
    def __init__(self, exp_name, run_name, db_path=None, **kwargs):
        self.tracking_uri = f"sqlite:///{'mlruns.db' if db_path is None else db_path}"
        mlflow.set_tracking_uri(uri=self.tracking_uri)
        mlflow.enable_system_metrics_logging()
        self.exp_name = exp_name
        self.run_name = run_name

        client = mlflow.MlflowClient(**kwargs)
        try:
            _logger.info(f"Create a new Experiment: {self.exp_name}.")
            self.exp_id = client.create_experiment(self.exp_name)
        except mlflow.MlflowException:
            _logger.info(f"Experiment {self.exp_name} already exists!")
            self.exp_id = client.get_experiment_by_name(self.exp_name).experiment_id
        self.run = mlflow.start_run(experiment_id=self.exp_id, run_name=self.run_name)

    def log_params(self, params):
        if isinstance(params, DictConfig):
            params = omegaconf_to_dict(params)
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        mlflow.log_metrics(metrics, step=step)

    def log_state_dict(self, model):
        mlflow.pytorch.log_state_dict(model.cpu().state_dict(), 'weights/model')

    def log_model(self, model):
        mlflow.pytorch.log_model(model, 'models')

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def log_artifacts(self, local_dir: str):
        mlflow.log_artifacts(local_dir)

    def log_figure(self, object: Union["numpy.ndarray", "PIL.Image.Image", "matplotlib.figure.Figure"], artifact_path: str):
        if isinstance(object, matplotlib.figure.Figure):
            mlflow.log_figure(object, artifact_path)
        else:
            mlflow.log_image(object, artifact_path)

    def finish(self):
        _logger.info(f"Experiment {self.exp_name}/{self.run_name} finish.")
        mlflow.end_run()

    def load_model_from_alias(self, model_name, alias):
        return mlflow.pytorch.load_model(f'models:/{model_name}@{alias}', map_location='cpu')

