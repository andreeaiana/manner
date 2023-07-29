from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from manner.data.components.mind_batch import MINDRecBatch
from manner.data.components.adressa_dataframe import AdressaDataFrame
from transformers import PreTrainedTokenizer


class AdressaRecDatasetTrain(AdressaDataFrame):
    def __init__(
        self,
        news: pd.DataFrame,
        behaviors: pd.DataFrame,
        max_history_length: int,
        neg_sampling_ratio: int,
    ) -> None:
        self.news = news
        self.behaviors = behaviors
        self.max_history_length = max_history_length
        self.neg_sampling_ratio = neg_sampling_ratio

    def __getitem__(self, idx: Any) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
        bhv = self.behaviors.iloc[idx]

        user = np.array([int(bhv["user"])])
        history = np.array(bhv["history"])[: self.max_history_length]
        candidates = np.array(bhv["candidates"])
        labels = np.array(bhv["labels"])

        candidates, labels = self._sample_candidates(candidates, labels)

        history = self.news.loc[history]
        candidates = self.news.loc[candidates]

        return user, history, candidates, labels

    def __len__(self) -> int:
        return len(self.behaviors)

    def _sample_candidates(self, candidates: np.ndarray, labels: np.ndarray) -> Tuple[List, List]:
        """Negative sampling of news candidates.

        Args:
            candidates (np.array): Candidate news.
            labels (np.array): Labels of candidate news.

        Returns:
            - List: Sampled candidates.
            - np.array: Bool labels of sampled candidates (e.g. True if candidate was clicked, False otherwise)
        """
        pos_ids = np.where(labels == 1)[0]
        neg_ids = np.array([]).astype(int)

        # sample with replacement when the candidates set is smaller than the negative sampling ratio
        replace_flag = (
            True
            if (self.neg_sampling_ratio * len(pos_ids) > len(labels) - len(pos_ids))
            else False
        )

        # do negative sampling with the specified ratio
        neg_ids = np.random.choice(
            np.random.permutation(np.where(labels == 0)[0]),
            self.neg_sampling_ratio * len(pos_ids),
            replace=replace_flag,
        )

        indices = np.concatenate((pos_ids, neg_ids))
        indices = np.random.permutation(indices)
        candidates = candidates[indices]
        labels = labels[indices]

        return candidates, labels


class AdressaRecDatasetTest(AdressaDataFrame):
    def __init__(
        self, news: pd.DataFrame, behaviors: pd.DataFrame, max_history_length: int
    ) -> None:
        self.news = news
        self.behaviors = behaviors
        self.max_history_length = max_history_length

    def __getitem__(self, idx: Any) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
        bhv = self.behaviors.iloc[idx]

        user = np.array([int(bhv["user"])])
        history = np.array(bhv["history"])[: self.max_history_length]
        candidates = np.array(bhv["candidates"])
        labels = np.array(bhv["labels"])

        history = self.news.loc[history]
        candidates = self.news.loc[candidates]

        return user, history, candidates, labels

    def __len__(self) -> int:
        return len(self.behaviors)


@dataclass
class AdressaCollate:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer 
        ) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch) -> MINDRecBatch:
        users, histories, candidates, labels = zip(*batch)

        batch_hist = _make_batch_assignees(histories)
        batch_cand = _make_batch_assignees(candidates)

        x_hist = self._tokenize_df(pd.concat(histories))
        x_cand = self._tokenize_df(pd.concat(candidates))
        labels = torch.from_numpy(np.concatenate(labels)).float()
        users = torch.from_numpy(np.concatenate(users)).long()

        return MINDRecBatch(
            batch_hist=batch_hist,
            batch_cand=batch_cand,
            x_hist=x_hist,
            x_cand=x_cand,
            labels=labels,
            users=users,
        )

    def _tokenize(self, x: List[str]):
        return self.tokenizer(
            x, return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True
        )

    def _tokenize_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "text": self._tokenize(df['title'].values.tolist()),
            "category": torch.from_numpy(df["category_label"].values).long(),
            "sentiment": torch.from_numpy(df["sentiment_label"].values).long(),
            "sentiment_score": torch.from_numpy(df["sentiment_score"].values).float(),
        }


def _make_batch_assignees(items: Sequence[Sequence[Any]]) -> torch.Tensor:
    sizes = torch.tensor([len(x) for x in items])
    batch = torch.repeat_interleave(torch.arange(len(items)), sizes)
    return batch
