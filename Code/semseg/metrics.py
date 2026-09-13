"""Per-image metrics and Tversky loss, with explicit empty-mask convention."""
import torch


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, eps=1e-6):
    probabilities = torch.sigmoid(logits)
    axes = (1, 2, 3)
    tp = (probabilities * targets).sum(dim=axes)
    fp = (probabilities * (1 - targets)).sum(dim=axes)
    fn = ((1 - probabilities) * targets).sum(dim=axes)
    return 1 - ((tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def binary_metrics(prediction, target, eps=1e-6):
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = int((prediction & target).sum())
    size = int(prediction.sum()) + int(target.sum())
    union = int((prediction | target).sum())
    return (2 * intersection + eps) / (size + eps), (intersection + eps) / (union + eps)
