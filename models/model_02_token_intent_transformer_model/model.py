from transformers import AutoModel
import torch.nn as nn

from utils.schemas import Model02LogitOutput

class Model02BankingNLUTransformerModel(nn.Module):

    def __init__(
        self,
        model_name:str,
        intent_count:int,
        entity_count:int,
        dropout:float=0.1
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.intent_head = nn.Linear(
            hidden,
            intent_count
        )

        self.entity_head = nn.Linear(
            hidden,
            entity_count
        )

    def forward(self, input_ids, attention_mask):

        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_state = self.dropout(
            output.last_hidden_state
        )

        intent_logits = self.intent_head(
            hidden_state
        )

        entity_logits = self.entity_head(
            hidden_state
        )

        return Model02LogitOutput(
            intent_logits=intent_logits,
            entity_logits=entity_logits
        )