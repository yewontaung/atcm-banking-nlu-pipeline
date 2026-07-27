from torch import Tensor
import torch.nn as nn
from transformers import AutoModel

from utils.schemas import TransformerModelOutput

class BankingNLUTransformerModel(nn.Module):

    def __init__(
        self, 
        model_name:str, 
        intent_count:int,
        entity_count:int,
        dropout:float = 0.1):

        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.intent_head = nn.Linear(hidden_size, intent_count)
        self.entity_head = nn.Linear(hidden_size, entity_count)

    def forward(self, input_ids:Tensor, attention_mask:Tensor) -> TransformerModelOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = outputs.last_hidden_state
        cls_embedding = hidden_state[:, 0]
        cls_embedding = self.dropout(cls_embedding)

        token_embedding = self.dropout(hidden_state)

        intent_logits = self.intent_head(cls_embedding)
        entity_logits = self.entity_head(token_embedding)

        return TransformerModelOutput(
            intent_logits=intent_logits,
            entity_logits=entity_logits,
        )