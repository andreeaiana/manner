import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from manner.models.components.attention import AdditiveAttention, DenseAttention


class NAMLUserEncoder(nn.Module):
    def __init__(self, news_embedding_dim: int, query_vector_dim: int) -> None:
        super().__init__()

        self.additive_attention = AdditiveAttention(
            input_dim=news_embedding_dim, query_dim=query_vector_dim
        )

    def forward(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        # batch_size, num_clicked_news_per_user, news_embedding_dim -> batch_size, news_embedding_dim
        user_vector = self.additive_attention(clicked_news_vector)

        return user_vector


class NRMSUserEncoder(nn.Module):
    def __init__(
        self, news_embedding_dim: int, num_attention_heads: int, query_vector_dim: int
    ) -> None:
        super().__init__()

        self.multihead_attention = nn.MultiheadAttention(news_embedding_dim, num_attention_heads)
        self.additive_attention = AdditiveAttention(news_embedding_dim, query_vector_dim)

    def forward(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        # batch_size, num_clicked_news_user, news_embeding_dim
        user_vector, _ = self.multihead_attention(
            clicked_news_vector, clicked_news_vector, clicked_news_vector
        )

        # batch_size, news_embeding_dim
        user_vector = self.additive_attention(user_vector)

        return user_vector


class LSTURUserEncoder(nn.Module):
    def __init__(
        self,
        num_users: int,
        input_dim: int,
        user_masking_probability: float,
        long_short_term_method: str,
    ) -> None:
        super().__init__()

        assert long_short_term_method in ["ini", "con"]
        self.long_short_term_method = long_short_term_method

        self.long_term_user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=input_dim
            if self.long_short_term_method == "ini"
            else int(input_dim * 0.5),
            padding_idx=0,
        )
        self.dropout = nn.Dropout2d(p=user_masking_probability)
        self.gru = nn.GRU(
            input_dim, input_dim if self.long_short_term_method == "ini" else int(input_dim * 0.5)
        )

    def forward(
        self, user: torch.Tensor, clicked_news_vector: torch.Tensor, hist_size: torch.Tensor
    ) -> torch.Tensor:
        # long-term user representation
        user_vector = self.long_term_user_embedding(user).unsqueeze(dim=0)
        user_vector = self.dropout(user_vector)

        # short-term user representation
        packed_clicked_news_vector = pack_padded_sequence(
            input=clicked_news_vector,
            lengths=hist_size.cpu().int(),
            batch_first=True,
            enforce_sorted=False,
        )
        if self.long_short_term_method == "ini":
            _, last_hidden = self.gru(packed_clicked_news_vector, user_vector)
            return last_hidden.squeeze(dim=0)
        else:
            _, last_hidden = self.gru(packed_clicked_news_vector)
            return torch.cat((last_hidden.squeeze(dim=0), user_vector.squeeze(dim=0)), dim=1)


class CAUMUserEncoder(nn.Module):
    # Adapted from https://github.com/taoqi98/CAUM/blob/main/Code/Models.py
    def __init__(
            self,
            news_vector_dim: int,
            num_filters: int,
            dense_att_hidden_dim1: int,
            dense_att_hidden_dim2: int,
            user_vector_dim: int,
            num_attention_heads: int,
            dropout_probability: float
            ) -> None:
        super().__init__()

        self.dropout1 = nn.Dropout(p=dropout_probability)
        self.dropout2 = nn.Dropout(p=dropout_probability)
        self.dropout3 = nn.Dropout(p=dropout_probability)

        self.linear1 = nn.Linear(news_vector_dim * 4, num_filters)
        self.linear2 = nn.Linear(news_vector_dim * 2, user_vector_dim)
        self.linear3 = nn.Linear(num_filters + user_vector_dim, user_vector_dim)

        self.dense_att = DenseAttention(
                input_dim=user_vector_dim * 2,
                hidden_dim1=dense_att_hidden_dim1,
                hidden_dim2=dense_att_hidden_dim2
                )
        self.multihead_attention = nn.MultiheadAttention(user_vector_dim, num_attention_heads)

    def forward(self, clicked_news_vector: torch.Tensor, cand_news_vector: torch.Tensor) -> torch.Tensor:
        cand_news_vector = self.dropout1(cand_news_vector)
        clicked_news_vector = self.dropout2(clicked_news_vector)

        repeated_cand_news_vector = cand_news_vector.unsqueeze(dim=1).repeat(1, clicked_news_vector.shape[1], 1)

        # candi-cnn
        clicked_news_left = torch.cat(
                [clicked_news_vector[:, -1:, :], clicked_news_vector[:, :-1, :]],
                dim=-2
                )
        clicked_news_right =  torch.cat(
                [clicked_news_vector[:, 1:, :], clicked_news_vector[:, :1, :]],
                dim=-2
                )
        clicked_news_cnn = torch.cat(
                [
                    clicked_news_left, 
                    clicked_news_vector, 
                    clicked_news_right, 
                    repeated_cand_news_vector
                    ],
                dim=-1
                )

        clicked_news_cnn = self.linear1(clicked_news_cnn)

        # candi-selfatt
        clicked_news = torch.cat(
                [repeated_cand_news_vector, clicked_news_vector],
                dim=-1) 
        clicked_news = self.linear2(clicked_news)
        clicked_news_self, _ = self.multihead_attention(
                clicked_news, 
                clicked_news, 
                clicked_news
                )

        clicked_news_all = torch.cat(
                [clicked_news_cnn, clicked_news_self], 
                dim=-1
                )
        clicked_news_all = self.dropout3(clicked_news_all)
        clicked_news_all = self.linear3(clicked_news_all)

        # candi-att
        attention_vector = torch.cat(
                [clicked_news_all, repeated_cand_news_vector],
                dim=-1)
        attention_score = self.dense_att(attention_vector)
        attention_score = attention_score.squeeze(dim=-1)
        attention_score = F.softmax(attention_score, dim=-1)

        user_vector = torch.bmm(attention_score.unsqueeze(dim=1), clicked_news_all).squeeze(dim=1)

        scores = torch.bmm(cand_news_vector.unsqueeze(dim=1), user_vector.unsqueeze(dim=-1)).flatten()

        return scores


class MINSUserEncoder(nn.Module):
    def __init__(
            self,
            news_embedding_dim: int,
            query_vector_dim: int,
            num_filters: int,
            num_gru_channels: int
            ) -> None:
        super().__init__()
        
        self.num_gru_channels = num_gru_channels

        self.multihead_attention = nn.MultiheadAttention(
                embed_dim=news_embedding_dim,
                num_heads=num_gru_channels
                )
        self.additive_attention = AdditiveAttention(
                input_dim=news_embedding_dim,
                query_dim=query_vector_dim
                )

        assert num_filters % num_gru_channels == 0
        self.gru = nn.GRU(
                int(num_filters / num_gru_channels),
                int(num_filters / num_gru_channels)
                )
        self.multi_channel_gru = nn.ModuleList(
                [self.gru for _ in range(num_gru_channels)]
                )

    def forward(self, clicked_news_vector: torch.Tensor, hist_size: torch.Tensor) -> torch.Tensor:
        # batch_size, hist_size, news_embedding_dim
        multihead_user_vector, _ = self.multihead_attention(
                clicked_news_vector, clicked_news_vector, clicked_news_vector
                )

        # batch_size, hist_size, news_embedding_dim / num_gru_channels
        user_vector_channels = torch.chunk(
                input=multihead_user_vector,
                chunks=self.num_gru_channels,
                dim=2
                )
        channels = []
        for n, gru in zip(range(self.num_gru_channels), self.multi_channel_gru):
            packed_clicked_news_vector = pack_padded_sequence(
                input=user_vector_channels[n],
                lengths=hist_size.cpu().int(),
                batch_first=True,
                enforce_sorted=False
                )

            # 1, batch_size, num_filters / num_gru_channels
            _, last_hidden = gru(packed_clicked_news_vector)

            channels.append(last_hidden)

        # batch_size, 1, news_embedding_dim
        multi_channel_vector = torch.cat(channels, dim=2).transpose(0, 1)

        # batch_size, news_embedding_dim
        user_vector = self.additive_attention(multi_channel_vector)

        return user_vector

