from typing import Optional

import torch
import torch.nn.functional as F
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs


def diversity(
    preds: torch.Tensor, target: torch.Tensor, num_classes: int, k: Optional[int] = None
) -> torch.Tensor:
    preds, target = _check_retrieval_functional_inputs(preds, target, allow_non_binary_target=True)

    k = preds.shape[-1] if k is None else k

    if not (isinstance(k, int) and k > 0):
        raise ValueError("`k` has to be a positive integer or None")

    sorted_target = target[torch.argsort(preds, dim=-1, descending=True)][:k]
    target_count = torch.bincount(sorted_target)
    padded_target_count = F.pad(target_count, pad=(0, num_classes - target_count.shape[0]))
    target_prob = padded_target_count / padded_target_count.shape[0]
    target_distribution = torch.distributions.Categorical(target_prob)

    diversity = torch.div(
        target_distribution.entropy(), torch.log(torch.tensor(num_classes, device=preds.device))
    )

    return diversity


def personalization(
    preds: torch.Tensor,
    predicted_aspects: torch.Tensor,
    target_aspects: torch.Tensor,
    num_classes: int,
    k: Optional[int] = None,
) -> torch.Tensor:
    preds, predicted_aspects = _check_retrieval_functional_inputs(
        preds, predicted_aspects, allow_non_binary_target=True
    )

    k = preds.shape[-1] if k is None else k

    if not (isinstance(k, int) and k > 0):
        raise ValueError("`k` has to be a positive integer or None")

    sorted_predicted_aspects = predicted_aspects[torch.argsort(preds, dim=-1, descending=True)][:k]
    predicted_aspects_count = torch.bincount(sorted_predicted_aspects)
    padded_predicted_aspects_count = F.pad(
        predicted_aspects_count, pad=(0, num_classes - predicted_aspects_count.shape[0])
    )

    target_aspects_count = torch.bincount(target_aspects)
    padded_target_aspects_count = F.pad(
        target_aspects_count, pad=(0, num_classes - target_aspects_count.shape[0])
    )

    personalization = generalized_jaccard(
        padded_predicted_aspects_count, padded_target_aspects_count
    )

    return personalization


def generalized_jaccard(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape

    jaccard = torch.min(pred, target).sum(dim=0) / torch.max(pred, target).sum(dim=0)

    return jaccard


def harmonic_mean(scores: torch.Tensor) -> torch.Tensor:
    weights = torch.ones(scores.shape, device=scores.device)
    harmonic_mean = torch.div(torch.sum(weights), torch.sum(torch.div(weights, scores)))

    return harmonic_mean
