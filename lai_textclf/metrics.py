from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from torchmetrics import MetricCollection


def get_default_clf_metrics(num_classes: int):
    return MetricCollection(
        MulticlassPrecision(num_classes),
        MulticlassRecall(num_classes),
        MulticlassF1Score(num_classes),
        MulticlassAccuracy(num_classes),
    )
