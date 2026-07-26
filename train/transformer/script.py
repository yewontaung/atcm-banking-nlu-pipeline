import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader


from dataprocessors.encoders import (
    IntentEncoder,
    EntityEncoder
)

from dataprocessors.preprocessors import (
    DataPreProcessor
)

from dataprocessors.tokenizers import (
    TokenizationProcessor
)


from dataloader.dataset import (
    NLUDataset
)

from dataloader.collator import (
    NLUCollator
)


from models.transformer_model import (
    BankingNLUTransformerModel
)


from train.transformer.loss import (
    TransformerNLULoss
)
from utils import env



DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================
# Create encoders
# ==========================
intent_encoder = IntentEncoder.from_file("./metadata/intents.json")
entity_encoder = EntityEncoder.from_file("./metadata/entities.json")
print(
    "Intent labels:",
    intent_encoder.label_to_id
)

print(
    "NER labels:",
    entity_encoder.label_to_id
)

# ==========================
# Process dataset
# ==========================

processor = DataPreProcessor()
processed_data = processor.process_file(f"{env.TRAINING_FILE}")
print(
    "Processed:",
    len(processed_data)
)

# ==========================
# Tokenization
# ==========================


tokenizer_processor = TokenizationProcessor(
    model_name="xlm-roberta-base",
    intent_encoder=intent_encoder,
    entity_encoder=entity_encoder
)



tokenized_data = [tokenizer_processor.process(item) for item in processed_data]

print(
    "Tokenized:",
    len(tokenized_data)
)


# ==========================
# Dataset
# ==========================

dataset = NLUDataset(samples=tokenized_data)

# ==========================
# DataLoader
# ==========================

collator = NLUCollator(
    pad_token_id=tokenizer_processor.tokenizer.pad_token_id
)

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collator
)

# ==========================
# Model
# ==========================

model = BankingNLUTransformerModel(
    model_name="xlm-roberta-base",
    intent_count=intent_encoder.no_of_lables,
    entity_count=entity_encoder.no_of_entities,
)

model.to(DEVICE)

# ==========================
# Optimizer
# ==========================

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

# ==========================
# Loss
# ==========================


loss_fn = TransformerNLULoss()

# ==========================
# Training
# ==========================

EPOCHS = int(env.EPOCHS)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = (
            batch["input_ids"]
            .to(DEVICE)
        )
        attention_mask = (
            batch["attention_mask"]
            .to(DEVICE)
        )
        intent_labels = (
            batch["intent_labels"]
            .float()
            .to(DEVICE)
        )
        entity_labels = (
            batch["ner_labels"]
            .long()
            .to(DEVICE)
        )

        optimizer.zero_grad()

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        losses = loss_fn(
            output,
            intent_labels,
            entity_labels
        )

        loss = losses["loss"]
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    avg_loss = (
        total_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch+1}: {avg_loss}"
    )

# ==========================
# Save
# ==========================

torch.save(
    {
        "model": model.state_dict(),
        "intent_encoder": intent_encoder.label_to_id,
        "entity_encoder": entity_encoder.label_to_id
    },
    f"{env.CHECKPOINT_PATH}/{env.SAVED_MODEL_NAME_PREFIX}_{env.TRAINING_DATASIZE}.pt"
)

print(
    "Training complete"
)