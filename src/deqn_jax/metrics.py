"""Metric logging backends for training monitoring.

Supports TensorBoard (via tensorboardX) and Weights & Biases.
All backends are optional — NullLogger is used when nothing is configured.
"""

from typing import Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MetricLogger(Protocol):
    """Protocol for metric logging backends."""

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None: ...
    def log_histograms(self, arrays: Dict[str, np.ndarray], step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class NullLogger:
    """No-op logger used when no backend is configured."""

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        pass

    def log_histograms(self, arrays: Dict[str, np.ndarray], step: int) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger:
    """TensorBoard logging via tensorboardX."""

    def __init__(self, log_dir: str):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError(
                "tensorboardX required for TensorBoard logging. "
                "Install with: uv pip install tensorboardX"
            )
        self._writer = SummaryWriter(log_dir)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)

    def log_histograms(self, arrays: Dict[str, np.ndarray], step: int) -> None:
        for key, arr in arrays.items():
            self._writer.add_histogram(key, arr, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class WandbLogger:
    """Weights & Biases logging."""

    def __init__(self, project: str, config: Optional[Dict] = None):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb required for W&B logging. "
                "Install with: uv pip install wandb"
            )
        wandb.init(project=project, config=config)
        self._wandb = wandb

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def log_histograms(self, arrays: Dict[str, np.ndarray], step: int) -> None:
        histograms = {k: self._wandb.Histogram(v) for k, v in arrays.items()}
        self._wandb.log(histograms, step=step)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._wandb.finish()


class CompositeLogger:
    """Forwards to multiple logging backends."""

    def __init__(self, loggers: List[MetricLogger]):
        self._loggers = loggers

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        for logger in self._loggers:
            logger.log_scalars(metrics, step)

    def log_histograms(self, arrays: Dict[str, np.ndarray], step: int) -> None:
        for logger in self._loggers:
            logger.log_histograms(arrays, step)

    def flush(self) -> None:
        for logger in self._loggers:
            logger.flush()

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()


def create_logger(
    tensorboard_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
) -> MetricLogger:
    """Create a metric logger from configuration.

    Returns NullLogger if nothing is configured, a single backend logger
    if one is specified, or a CompositeLogger if multiple are specified.
    """
    loggers: List[MetricLogger] = []

    if tensorboard_dir is not None:
        loggers.append(TensorBoardLogger(tensorboard_dir))

    if wandb_project is not None:
        loggers.append(WandbLogger(wandb_project, config=wandb_config))

    if len(loggers) == 0:
        return NullLogger()
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return CompositeLogger(loggers)
