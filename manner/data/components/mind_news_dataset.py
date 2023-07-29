import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from manner.data.components.mind_batch import MINDNewsBatch
from manner.data.components.mind_dataframe import MINDDataFrame
from transformers import PreTrainedTokenizer


class MINDNewsDataset(MINDDataFrame):
    def __init__(self, news: pd.DataFrame, behaviors: pd.DataFrame, aspect: str) -> None:
        news_ids = np.array(
            list(
                set(
                    list(itertools.chain.from_iterable(behaviors.history))
                    + list(itertools.chain.from_iterable(behaviors.candidates))
                )
            )
        )

        self.news = news.loc[news_ids]
        self.labels = np.array(self.news[aspect + "_label"])

    def __getitem__(self, idx: Any) -> Tuple[pd.DataFrame, int]:
        news = self.news.iloc[[idx]]
        label = self.labels[idx]

        return news, label

    def __len__(self) -> int:
        return len(self.news)


@dataclass
class MINDCollate:
    def __init__(
        self, tokenizer: PreTrainedTokenizer, text_aspects: List[str], entity_aspects: List[str]
    ) -> None:
        self.tokenizer = tokenizer
        self.text_aspects = text_aspects
        self.entity_aspects = entity_aspects

    def __call__(self, batch) -> MINDNewsBatch:
        news, labels = zip(*batch)

        transformed_news = self._tokenize_df(pd.concat(news))
        labels = torch.tensor(labels).long()

        return MINDNewsBatch(news=transformed_news, labels=labels)

    def _tokenize(self, x: List[str]):
        return self.tokenizer(
            x, return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True
        )

    def _tokenize_entities(self, x: List[int]) -> torch.Tensor:
        max_len = max([len(item) for item in x])
        x_vector_padded = [
            F.pad(torch.tensor(item), (0, max_len - len(item)), "constant", 0) for item in x
        ]
        return torch.vstack([item for item in x_vector_padded]).long()

    def _tokenize_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(self.entity_aspects) == 2:
            entities = [
                [*l1, *l2]
                for (l1, l2) in list(
                    zip(
                        df["title_entities"].values.tolist(),
                        df["abstract_entities"].values.tolist(),
                    )
                )
            ]
        else:
            entities = df["title_entities"].values.tolist()

        return {
            "text": self._tokenize(df[self.text_aspects].values.tolist())
            if len(self.text_aspects) == 2
            else self._tokenize(df[self.text_aspects[0]].values.tolist()),
            "entities": self._tokenize_entities(entities),
        }
