from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassAccuracy, MulticlassF1Score,
                                         MulticlassPrecision, MulticlassRecall)


def get_default_clf_metrics(num_classes: int):
    return MetricCollection(
        MulticlassPrecision(num_classes),
        MulticlassRecall(num_classes),
        MulticlassF1Score(num_classes),
        MulticlassAccuracy(num_classes),
    )
