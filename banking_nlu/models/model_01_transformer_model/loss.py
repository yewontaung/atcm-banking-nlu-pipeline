import torch.nn as nn

from banking_nlu.utils.schemas import Model01LogitOutput

class TransformerNLULoss:

    def __init__(self):
        self.intent_loss = nn.BCEWithLogitsLoss()
        self.entity_loss = nn.CrossEntropyLoss()

    def __call__(self, outputs:Model01LogitOutput, intent_labels, entity_labels):
        intent_loss = self.intent_loss(
            outputs.intent_logits,
            intent_labels.float()
        )

        entity_loss = self.entity_loss(
            outputs.entity_logits.view(
                -1, outputs.entity_logits.shape[-1]
            ),
            entity_labels.view(-1)
        )

        loss = intent_loss + entity_loss

        return {
            "loss": loss,
            "intent_loss": intent_loss,
            "entity_loss": entity_loss
        }