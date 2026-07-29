from torch import Tensor
import torch.nn as nn

from utils.schemas import Model02LogitOutput


class Model02TokenIntentEntityLoss(nn.Module):

    def __init__(
        self,
        intent_weight: float = 1.0,
        entity_weight: float = 1.0,
        ignore_index: int = -100
    ):
        super().__init__()

        self.intent_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )

        self.entity_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )

        self.intent_weight = intent_weight
        self.entity_weight = entity_weight


    def forward(
        self,
        output:Model02LogitOutput,
        intent_labels:Tensor,
        entity_labels:Tensor
    ):

        """
        intent_logits:
            (batch, seq_len, intent_count)

        entity_logits:
            (batch, seq_len, entity_count)

        labels:
            (batch, seq_len)
        """

        intent_logits = output.intent_logits
        entity_logits = output.entity_logits


        intent_loss = self.intent_loss(
            intent_logits.reshape(-1, intent_logits.shape[-1]),
            intent_labels.reshape(-1).long()
        )


        entity_loss = self.entity_loss(
            entity_logits.reshape(-1, entity_logits.shape[-1]),
            entity_labels.reshape(-1).long()
        )


        total_loss = (
            self.intent_weight * intent_loss
            +
            self.entity_weight * entity_loss
        )


        return {
            "loss": total_loss,
            "intent_loss": intent_loss,
            "entity_loss": entity_loss
        }