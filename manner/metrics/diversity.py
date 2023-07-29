from typing import Any, Optional

import torch
from manner.metrics.functional import diversity
from torchmetrics.retrieval.base import RetrievalMetric


class Diversity(RetrievalMetric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            empty_target_action=empty_target_action,
            ignore_index=ignore_index,
            **kwargs,
        )

        if (k is not None) and not (isinstance(k, int) and k > 0):
            raise ValueError("`k` has to be a positive integer or None")
        self.num_classes = num_classes
        self.k = k
        self.allow_non_binary_target = True

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return diversity(preds, target, self.num_classes, k=self.k)
