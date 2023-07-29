# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_retrieval_inputs
from torchmetrics.utilities.data import _flexible_bincount, dim_zero_cat


class CustomRetrievalMetric(Metric, ABC):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    cand_indexes: List[torch.Tensor]
    hist_indexes: List[torch.Tensor]
    preds: List[torch.Tensor]
    candidate_aspects: List[torch.Tensor]
    clicked_aspects: List[torch.Tensor]

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.allow_non_binary_target = True

        empty_target_action_options = ("error", "skip", "neg", "pos")
        if empty_target_action not in empty_target_action_options:
            raise ValueError(
                f"Argument `empty_target_action` received a wrong value `{empty_target_action}`."
            )

        self.empty_target_action = empty_target_action

        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ValueError("Argument `ignore_index` must be an integer or None.")

        self.ignore_index = ignore_index

        self.add_state("cand_indexes", default=[], dist_reduce_fx=None)
        self.add_state("hist_indexes", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("candidate_aspects", default=[], dist_reduce_fx=None)
        self.add_state("clicked_aspects", default=[], dist_reduce_fx=None)

    def update(
        self,
        preds: torch.Tensor,
        candidate_aspects: torch.Tensor,
        clicked_aspects: torch.Tensor,
        cand_indexes: torch.Tensor,
        hist_indexes: torch.Tensor,
    ) -> None:
        """Check shape, check and convert dtypes, flatten and add to accumulators."""
        if cand_indexes is None:
            raise ValueError("Argument `cand_indexes` cannot be None")
        if hist_indexes is None:
            raise ValueError("Argument `hist_indexes` cannot be None")

        cand_indexes, preds, candidate_aspects = _check_retrieval_inputs(
            cand_indexes,
            preds,
            candidate_aspects,
            allow_non_binary_target=self.allow_non_binary_target,
            ignore_index=self.ignore_index,
        )
        hist_indexes = hist_indexes.long().flatten()
        clicked_aspects = clicked_aspects.long().flatten()

        self.cand_indexes.append(cand_indexes)
        self.hist_indexes.append(hist_indexes)
        self.preds.append(preds)
        self.candidate_aspects.append(candidate_aspects)
        self.clicked_aspects.append(clicked_aspects)

    def compute(self) -> torch.Tensor:
        cand_indexes = dim_zero_cat(self.cand_indexes)
        hist_indexes = dim_zero_cat(self.hist_indexes)
        preds = dim_zero_cat(self.preds)
        candidate_aspects = dim_zero_cat(self.candidate_aspects)
        clicked_aspects = dim_zero_cat(self.clicked_aspects)

        cand_indexes, cand_indices = torch.sort(cand_indexes)
        hist_indexes, hist_indices = torch.sort(hist_indexes)
        preds = preds[cand_indices]
        candidate_aspects = candidate_aspects[cand_indices]
        clicked_aspects = clicked_aspects[hist_indices]

        cand_split_sizes = _flexible_bincount(cand_indexes).detach().cpu().tolist()
        hist_split_sizes = _flexible_bincount(hist_indexes).detach().cpu().tolist()

        res = []
        for mini_preds, mini_candidate_aspects, mini_clicked_aspects in zip(
            torch.split(preds, cand_split_sizes, dim=0),
            torch.split(candidate_aspects, cand_split_sizes, dim=0),
            torch.split(clicked_aspects, hist_split_sizes, dim=0),
        ):
            if not mini_candidate_aspects.sum():
                if self.empty_target_action == "error":
                    raise ValueError(
                        "`compute` method was provided with a query with no positive target."
                    )
                if self.empty_target_action == "pos":
                    res.append(torch.tensor(1.0))
                elif self.empty_target_action == "neg":
                    res.append(torch.tensor(0.0))
            else:
                # ensure list contains only float tensors
                res.append(self._metric(mini_preds, mini_candidate_aspects, mini_clicked_aspects))

        return (
            torch.stack([x.to(preds) for x in res]).mean() if res else torch.tensor(0.0).to(preds)
        )

    @abstractmethod
    def _metric(
        self, preds: torch.Tensor, candidate_aspects: torch.Tensor, clicked_aspects: torch.Tensor
    ) -> torch.Tensor:
        """Compute a metric over a predictions and target of a single group.

        This method should be overridden by subclasses.
        """
