from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


class SentimentAnnotator(nn.Module):
    def __init__(
            self, 
            sentiment_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            tokenizer_max_length: int  = 96
            ) -> None:
        
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
                sentiment_model,
                model_max_length=tokenizer_max_length
                )
        self.config = AutoConfig.from_pretrained(sentiment_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(sentiment_model)

    def forward(self, text: str) -> Tuple[str, float]:
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        logits = self.model(**encoded_input).logits
        scores = F.softmax(logits[0], dim=0).detach().numpy()

        label = self.config.id2label[scores.argmax()]
        if label == 'positive':
            sentiment_score = scores[scores.argmax()]
        elif label == 'negative':
            sentiment_score = -scores[scores.argmax()]
        else:
            sentiment_score = (1 - scores[scores.argmax()]) * (scores[-1] - scores[0])

        return (label, sentiment_score)

if __name__ == "__main__":
    _ = SentimentAnnotator()
