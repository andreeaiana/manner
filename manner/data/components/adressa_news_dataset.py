import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from manner.data.components.mind_batch import MINDNewsBatch
from manner.data.components.adressa_dataframe import AdressaDataFrame
from transformers import PreTrainedTokenizer


class AdressaNewsDataset(AdressaDataFrame):
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
class AdressaCollate:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer
        ) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch) -> MINDNewsBatch:
        news, labels = zip(*batch)

        transformed_news = self._tokenize_df(pd.concat(news))
        labels = torch.tensor(labels).long()

        return MINDNewsBatch(news=transformed_news, labels=labels)

    def _tokenize(self, x: List[str]):
        return self.tokenizer(
            x, return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True
        )

    def _tokenize_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "text": self._tokenize(df['title'].values.tolist())
        }
