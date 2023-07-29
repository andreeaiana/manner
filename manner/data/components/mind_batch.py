from typing import Any, Dict, Optional, TypedDict

import torch


class MINDRecBatch(TypedDict):
    batch_hist: torch.Tensor
    batch_cand: torch.Tensor
    x_hist: Dict[str, Any]
    x_cand: Dict[str, Any]
    labels: Optional[torch.Tensor]
    users: torch.Tensor


class MINDNewsBatch(TypedDict):
    news: Dict[str, Any]
    labels: torch.Tensor
