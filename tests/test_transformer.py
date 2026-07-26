import torch

from models.transformer_model import (
    BankingNLUTransformerModel
)


model = BankingNLUTransformerModel(
    model_name="xlm-roberta-base",
    intent_count=8,
    entity_count=11,
)

input_ids = torch.randint(
    low=0,
    high=1000,
    size=(2, 16)
)

attention_mask = torch.ones(
    (2, 16),
    dtype=torch.long
)

output = model(
    input_ids=input_ids,
    attention_mask=attention_mask
)


print(output.intent_logits.shape)

print(output.entity_logits.shape)