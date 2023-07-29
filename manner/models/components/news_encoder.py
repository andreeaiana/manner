from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BatchEncoding

from manner.models.components.attention import AdditiveAttention


class MannerTextEncoder(nn.Module):
    def __init__(
        self, 
        plm_model: str, 
        frozen_layers: List[int], 
        dropout_probability: float
        ) -> None:
        super().__init__()

        self.plm_model = AutoModel.from_pretrained(plm_model)
        self.dropout = nn.Dropout(p=dropout_probability)

        # freeze PLM layers
        for name, param in self.plm_model.base_model.named_parameters():
            for layer in frozen_layers:
                if "layer." + str(layer) + "." in name:
                    param.requires_grad = False

    def forward(self, tokenized_text: BatchEncoding) -> torch.Tensor:
        text_vector = self.plm_model(
            **tokenized_text,
            output_attentions=False,
            output_hidden_states=False,
        ).last_hidden_state[:, 0, :]
        text_vector = self.dropout(text_vector)

        return text_vector


class MannerEntityEncoder(nn.Module):
    def __init__(
        self,
        pretrained_embedding: nn.Embedding,
        embedding_dim: int,
        num_attention_heads: int,
        query_vector_dim: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        self.pretrained_embedding = pretrained_embedding
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_attention_heads
        )
        self.additive_attention = AdditiveAttention(
            input_dim=embedding_dim, query_dim=query_vector_dim
        )
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, entity_sequence: torch.Tensor) -> torch.Tensor:
        # batch_size, num_entities_text
        entity_vector = self.pretrained_embedding(entity_sequence)
        entity_vector = self.dropout(entity_vector)

        # batch_size, num_entities_text, embedding_dim
        entity_vector, _ = self.multihead_attention(entity_vector, entity_vector, entity_vector)
        entity_vector = self.dropout(entity_vector)

        # batch_size, embedding_dim
        entity_vector = self.additive_attention(entity_vector)

        return entity_vector


class MannerNewsEncoder(nn.Module):
    def __init__(
        self,
        plm_model: str,
        frozen_layers: List[int],
        dropout_probability: float,
        use_entities: bool,
        entity_embeddings: torch.Tensor,
        entity_embedding_dim: int,
        num_attention_heads: int,
        query_vector_dim: int,
        text_embedding_dim: int,
    ) -> None:
        super().__init__()

        self.text_encoder = MannerTextEncoder(
            plm_model=plm_model,
            frozen_layers=frozen_layers,
            dropout_probability=dropout_probability,
        )

        self.use_entities = use_entities
        if self.use_entities:
            pretrained_entity_embedding = nn.Embedding.from_pretrained(
                embeddings=torch.FloatTensor(entity_embeddings), 
                freeze=False, 
                padding_idx=0
            )
            self.entity_encoder = MannerEntityEncoder(
                pretrained_embedding=pretrained_entity_embedding,
                embedding_dim=entity_embedding_dim,
                num_attention_heads=num_attention_heads,
                query_vector_dim=query_vector_dim,
                dropout_probability=dropout_probability,
            )
            self.linear = nn.Linear(
                in_features=text_embedding_dim + entity_embedding_dim,
                out_features=text_embedding_dim,
            )

    def forward(self, news: Dict[str, Any]) -> torch.Tensor:
        # text embedding
        text_vector = self.text_encoder(news["text"])

        if self.use_entities:
            # entity embedding
            entity_vector = self.entity_encoder(news["entities"])

            # concat text and entity vectors
            news_vector = self.linear(torch.cat([text_vector, entity_vector], dim=-1))
        else:
            # only text vector
            news_vector = text_vector

        return news_vector


class PLMTextEncoder(nn.Module):
    def __init__(
        self,
        plm_model: str,
        frozen_layers: List[int],
        text_embedding_dim: int,
        num_attention_heads: int,
        query_vector_dim: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        self.plm_model = AutoModel.from_pretrained(plm_model)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=text_embedding_dim, num_heads=num_attention_heads
        )
        self.additive_attention = AdditiveAttention(
            input_dim=text_embedding_dim, query_dim=query_vector_dim
        )
        self.dropout = nn.Dropout(p=dropout_probability)

        # freeze PLM layers
        for name, param in self.plm_model.base_model.named_parameters():
            for layer in frozen_layers:
                if "layer." + str(layer) + "." in name:
                    param.requires_grad = False

    def forward(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        # batch_size, num_words_text
        text_vector = self.plm_model(**tokenized_text)[0]
        text_vector = self.dropout(text_vector)

        # batch_size, num_words_text, text_embedding_dim
        text_vector, _ = self.multihead_attention(text_vector, text_vector, text_vector)
        text_vector = self.dropout(text_vector)

        # batch_size, text_embedding_dim
        text_vector = self.additive_attention(text_vector)

        return text_vector


class NAMLCategoryEncoder(nn.Module):
    def __init__(
            self,
            num_categories: int,
            category_embedding_dim: int,
            category_output_dim: int
            ) -> None:

        super().__init__()

        self.category_embedding = nn.Embedding(
                num_embeddings=num_categories,
                embedding_dim=category_embedding_dim,
                padding_idx=0
                )
        self.linear = nn.Linear(category_embedding_dim, category_output_dim)

    def forward(self, category: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(self.category_embedding(category)))


class NAMLNewsEncoder(nn.Module):
    def __init__(
            self,
            plm_model: str,
            frozen_layers: List[int],
            text_embedding_dim: int,
            num_attention_heads: int,
            query_vector_dim: int,
            dropout_probability: float,
            num_categories: int,
            category_embedding_dim: int,
            ) -> None:

        super().__init__()

        self.text_encoder = PLMTextEncoder(
                plm_model=plm_model,
                frozen_layers=frozen_layers,
                text_embedding_dim=text_embedding_dim,
                num_attention_heads=num_attention_heads,
                query_vector_dim=query_vector_dim,
                dropout_probability=dropout_probability
                )

        self.category_encoder = NAMLCategoryEncoder(
                num_categories=num_categories,
                category_embedding_dim=category_embedding_dim,
                category_output_dim=text_embedding_dim
                )

        self.additive_attention = AdditiveAttention(text_embedding_dim, query_vector_dim)

    def forward(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_vector = self.text_encoder(news['text'])
        category_vector = self.category_encoder(news['category'])

        all_vectors = [text_vector] + [category_vector]
        final_news_vector = self.additive_attention(
                torch.stack(all_vectors, dim=1)
                )

        return final_news_vector


class LSTURCategoryEncoder(nn.Module):
    def __init__(
            self,
            num_categories: int,
            category_embedding_dim: int,
            ) -> None:

        super().__init__()

        self.category_embedding = nn.Embedding(
                num_embeddings=num_categories,
                embedding_dim=category_embedding_dim,
                padding_idx=0
                )

    def forward(self, category: torch.Tensor) -> torch.Tensor:
        return self.category_embedding(category)


class LSTURNewsEncoder(nn.Module):
    def __init__(
            self,
            plm_model: str,
            frozen_layers: List[int],
            text_embedding_dim: int,
            num_attention_heads: int,
            query_vector_dim: int,
            dropout_probability: float,
            num_categories: int,
            category_embedding_dim: int,
            ) -> None:

        super().__init__()

        self.text_encoder = PLMTextEncoder(
                plm_model=plm_model,
                frozen_layers=frozen_layers,
                text_embedding_dim=text_embedding_dim,
                num_attention_heads=num_attention_heads,
                query_vector_dim=query_vector_dim,
                dropout_probability=dropout_probability
                )

        self.category_encoder = LSTURCategoryEncoder(
                num_categories=num_categories,
                category_embedding_dim=category_embedding_dim
                )

    def forward(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_vector = self.text_encoder(news['text'])
        category_vector = self.category_encoder(news['category'])

        all_vectors = [text_vector] + [category_vector]
        final_news_vector = torch.cat(all_vectors, dim=1)

        return final_news_vector


class MINERNewsEncoder(nn.Module):
    def __init__(
            self, plm_model: str, frozen_layers: List[int], apply_reduce_dim: bool, text_embedding_dim: int, news_embedding_dim: int, dropout_probability: float
    ) -> None:
        super().__init__()

        self.plm_model = AutoModel.from_pretrained(plm_model)
        # freeze PLM layers
        for name, param in self.plm_model.base_model.named_parameters():
            for layer in frozen_layers:
                if "layer." + str(layer) + "." in name:
                    param.requires_grad = False

        self.apply_reduce_dim = apply_reduce_dim
        if self.apply_reduce_dim:
            self.reduce_dim = nn.Linear(
                    in_features=text_embedding_dim,
                    out_features=news_embedding_dim
                    )
            self.dropout = nn.Dropout(p=dropout_probability)
        
    def forward(self, tokenized_text: BatchEncoding) -> torch.Tensor:
        text_vector = self.plm_model(
            **tokenized_text,
            output_attentions=False,
            output_hidden_states=False,
        ).last_hidden_state[:, 0, :]
        if self.apply_reduce_dim:
            text_vector = self.reduce_dim(text_vector)
            text_vector = self.dropout(text_vector)

        return text_vector


class CAUMCategoryEncoder(nn.Module):
    def __init__(
            self,
            num_categories: int,
            category_embedding_dim: int,
            category_output_dim: int,
            dropout_probability: float
            ) -> None:

        super().__init__()

        self.category_embedding = nn.Embedding(
                num_embeddings=num_categories, 
                embedding_dim=category_embedding_dim,
                padding_idx=0
                )
        self.linear = nn.Linear(category_embedding_dim, category_output_dim)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, category: torch.Tensor) -> torch.Tensor:
        category_vector = self.category_embedding(category)
        category_vector = self.dropout(category_vector)
        category_vector = F.relu(self.linear(category_vector))

        return category_vector


class CAUMNewsEncoder(nn.Module):
    # Adapted from https://github.com/taoqi98/CAUM/blob/main/Code/Models.py
    def __init__(
            self,
            plm_model: str,
            frozen_layers: List[int],
            text_embedding_dim: int,
            text_num_attention_heads: int,
            query_vector_dim: int,
            dropout_probability: float,
            num_categories: int,
            category_embedding_dim: int,
            use_entities: bool, 
            entity_embeddings: nn.Embedding,
            entity_embedding_dim: int,
            entity_num_attention_heads: int,
            news_out_embedding_dim: int
            ) -> None:

        super().__init__()

        self.text_encoder = PLMTextEncoder(
                plm_model=plm_model,
                frozen_layers=frozen_layers,
                text_embedding_dim=text_embedding_dim,
                num_attention_heads=text_num_attention_heads,
                query_vector_dim=query_vector_dim,
                dropout_probability=dropout_probability
                )

        self.category_encoder = CAUMCategoryEncoder(
                num_categories=num_categories,
                category_embedding_dim=category_embedding_dim,
                category_output_dim=category_embedding_dim,
                dropout_probability=dropout_probability
                )

        self.use_entities = use_entities
        if self.use_entities:
            pretrained_entity_embedding = nn.Embedding.from_pretrained(
                embeddings=torch.FloatTensor(entity_embeddings), freeze=False, padding_idx=0
            )
            self.entity_encoder = MannerEntityEncoder(
                    pretrained_embedding=pretrained_entity_embedding,
                    embedding_dim=entity_embedding_dim,
                    num_attention_heads=entity_num_attention_heads,
                    query_vector_dim=query_vector_dim,
                    dropout_probability=dropout_probability
                    )
            linear_in_features = text_embedding_dim + entity_embedding_dim + category_embedding_dim
        else:
            linear_in_features = text_embedding_dim + category_embedding_dim
        
        self.linear = nn.Linear(
                in_features=linear_in_features,
                out_features=news_out_embedding_dim
                )

    def forward(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_vector = self.text_encoder(news['text'])
        category_vector = self.category_encoder(news['category'])

        if self.use_entities:
            entity_vector = self.entity_encoder(news['entities'])
            all_vectors = torch.cat(
                    [text_vector, entity_vector, category_vector],
                    dim=-1
                    )
        else:
            all_vectors = torch.cat(
                    [text_vector, category_vector],
                    dim=-1
                    )

        news_vector = self.linear(all_vectors)

        return news_vector

