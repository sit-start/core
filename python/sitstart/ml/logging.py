import torcheval.metrics
import torchmetrics
import torchmetrics.classification
from matplotlib.figure import Figure
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from sklearn.metrics import ConfusionMatrixDisplay

from sitstart.logging import get_logger
from sitstart.ml.metrics import MulticlassConfusionMatrix

logger = get_logger(__name__)

Metric = torcheval.metrics.Metric | torchmetrics.Metric

_torchmetrics_conf_mat_metrics = (
    torchmetrics.classification.MulticlassConfusionMatrix,
    torchmetrics.classification.BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
)
_torcheval_conf_mat_metrics = (
    torcheval.metrics.MulticlassConfusionMatrix,
    torcheval.metrics.BinaryConfusionMatrix,
)
_conf_mat_metrics = (*_torchmetrics_conf_mat_metrics, *_torcheval_conf_mat_metrics)


def is_multidim_metric(metric: Metric) -> bool:
    return isinstance(metric, _conf_mat_metrics)


def _get_conf_mat_fig(metric: Metric) -> Figure | None:
    # TODO: confirm axes are consistent for all supported types
    if isinstance(metric, _conf_mat_metrics):
        conf_mat = metric.compute()
        labels = getattr(metric, "labels", None)
        vis = ConfusionMatrixDisplay(conf_mat.cpu().numpy(), display_labels=labels)
        fig = vis.plot().figure_
        return fig


def log_multidim_metric(pl_logger: Logger, name: str, metric: Metric) -> None:
    # TODO: add support for additional metrics
    did_log = False

    conf_mat_fig = _get_conf_mat_fig(metric)
    if isinstance(pl_logger, TensorBoardLogger) and conf_mat_fig:
        pl_logger.experiment.add_figure(name, conf_mat_fig)
        did_log = True

    if not did_log:
        logger.warning(
            f"No logger found for multi-dim metric of type {type(metric)!r}. "
            f"Metric value:\n{metric.compute()}"
        )
