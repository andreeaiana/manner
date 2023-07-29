import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, query_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, query_dim)
        self.query = nn.Parameter(torch.empty(query_dim).uniform_(-0.1, 0.1))

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_vector (torch.Tensor): User tensor of shape (batch_size, hidden_dim, output_dim)

        Returns:
            torch.Tensor: User tensor of shape (batch_size, news_emb_dim).
        """
        # batch_size, hidden_dim, output_dim
        attention = torch.tanh(self.linear(input_vector))

        # batch_size, hidden_dim
        attention_weights = F.softmax(torch.matmul(attention, self.query), dim=1)

        # batch_size, output_dim
        weighted_input = torch.bmm(attention_weights.unsqueeze(dim=1), input_vector).squeeze(dim=1)

        return weighted_input


class PolyAttention(nn.Module):
    """
    Implementation of Poly attention scheme that extracts K attention vectors through K additive attentions.
    Adapted from https://github.com/duynguyen-0203/miner/blob/master/src/model/model.py.
    """
    def __init__(
            self,
            input_embed_dim: int,
            num_context_codes: int,
            context_code_dim: int
            ) -> None:
        """
        Args:
            in_embed_dim (int): The number of expected features in the input.
            num_context_codes (int): The number of attention vectors.
            context_code_dim (int): The number of features in a context code.
        """

        super().__init__()

        self.linear = nn.Linear(in_features=input_embed_dim, out_features=context_code_dim, bias=False)
        self.context_codes = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(num_context_codes, context_code_dim),
                    gain=nn.init.calculate_gain('tanh')
                    )
                )

    def forward(self, clicked_news_vector: torch.Tensor, attn_mask: torch.Tensor, bias: torch.Tensor = None):
        """
        Args:
            clicked_news_vector (torch.Tensor): (batch_size, hist_length, embed_dim)
            attn_mask (torch.Tensor): (batch_size, hist_length)
            bias (torch.Tensor): (batch_size, hist_length, num_candidates)

        Returns:
            torch.Tensor: (batch_size, num_context_codes, embed_dim)
        """
        projection = torch.tanh(self.linear(clicked_news_vector))
        
        if bias is None:
            weights = torch.matmul(projection, self.context_codes.T)
        else:
            bias = bias.mean(dim=2).unsqueeze(dim=2)
            weights = torch.matmul(projection, self.context_codes.T) + bias

        weights = weights.permute(0, 2, 1)
        weights = weights.masked_fill_(~attn_mask.unsqueeze(dim=1), 1e-30)
        weights = F.softmax(weights, dim=2)
        
        poly_news_vector = torch.matmul(weights, clicked_news_vector)

        return poly_news_vector


class TargetAwareAttention(nn.Module):
    """
    Implementation of the target-aware attention network. 
    Adapted from https://github.com/duynguyen-0203/miner/blob/master/src/model/model.py
    """
    def __init__(self, input_embed_dim: int) -> None:
        """
        Args:
            input_embed_dim (int): The number of features in the query and key vectors.
        """

        super().__init__()

        self.linear = nn.Linear(in_features=input_embed_dim, out_features=input_embed_dim, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): (batch_size, num_context_codes, input_embed_dim)
            key (torch.Tensor): (batch_size, num_candidates, input_embed_dim)
            value (torch.Tensor): (batch_size, num_candidates, num_context_codes)
        """
        projection = F.gelu(self.linear(query))
        weights = F.softmax(
                torch.matmul(key, projection.permute(0, 2, 1)),
                dim=2
                )
        outputs = torch.mul(weights, value).sum(dim=2)

        return outputs


class DenseAttention(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim1: int,
            hidden_dim2: int
            ) -> None:
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim1)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(hidden_dim2, 1)

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        transformed_vector = self.linear(input_vector)
        transformed_vector = self.tanh1(transformed_vector)
        transformed_vector = self.linear2(transformed_vector)
        transformed_vector = self.tanh2(transformed_vector)
        transformed_vector = self.linear3(transformed_vector)

        return transformed_vector
