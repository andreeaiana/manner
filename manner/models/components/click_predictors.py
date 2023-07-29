import torch
import torch.nn as nn


class DotProduct(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, clicked_news_vector: torch.Tensor, candidate_news_vector: torch.Tensor
    ) -> torch.Tensor:
        return torch.bmm(clicked_news_vector, candidate_news_vector).squeeze(1)
