import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from banking_nlu.dataprocessors.encoders.classification import (
    IntentClassificationEncoder
)

from banking_nlu.dataprocessors.encoders.bio import (
    BIOLabelEncoder
)


from banking_nlu.dataprocessors.preprocessors import (
    DataPreProcessor
)


from banking_nlu.dataprocessors.tokenizers import (
    TextTokenizer,
    Model01TransformerModelTokenizationProcessor
)


from banking_nlu.dataloader.dataset import (
    NLUDataset
)


from banking_nlu.dataloader.collator import (
    NLUCollator
)


from banking_nlu.models.model_01_transformer_model.model import (
    BankingNLUTransformerModel
)


from banking_nlu.models.model_01_transformer_model.loss import (
    TransformerNLULoss
)


from scripts.trainer import NLUModelTrainer

from banking_nlu.utils import env

from banking_nlu.utils.checkpoint import save_model



DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# ======================
# Encoders
# ======================

intent_encoder = IntentClassificationEncoder.from_file(
    env.INTENT_META_FILE
)

entity_encoder = BIOLabelEncoder.from_file(
    env.ENTITY_META_FILE
)



# ======================
# Data
# ======================

processor = DataPreProcessor()

train_processed = processor.process_file(
    env.TRAIN_JSON
)

validation_processed = processor.process_file(
    env.VALIDATE_JSON
)



tokenizer = TextTokenizer(
    AutoTokenizer.from_pretrained("xlm-roberta-base")
)


tokenizer_processor = (
    Model01TransformerModelTokenizationProcessor(
        tokenizer,
        intent_encoder,
        entity_encoder
    )
)


train_tokenized = [
    tokenizer_processor.process(item)
    for item in train_processed
]

validation_tokenized = [
    tokenizer_processor.process(item)
    for item in validation_processed
]


train_dataset = NLUDataset(train_tokenized)
validation_dataset = NLUDataset(validation_tokenized)


collator = NLUCollator(
    pad_token_id=tokenizer.tokenizer.pad_token_id,
    intent_mode="classification"
)



train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collator
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collator
)


# ======================
# Model
# ======================

model = BankingNLUTransformerModel(
    model_name="xlm-roberta-base",
    intent_count=intent_encoder.no_of_lables,
    entity_count=entity_encoder.no_of_labels
)


model.to(DEVICE)



optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)


loss_fn = TransformerNLULoss()



# ======================
# Train
# ======================

trainer = NLUModelTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=DEVICE
)

best_val_loss = float("inf")
best_epoch = 0

print()
print(
    f"| {'Epoch':^7} | "
    f"{'Train Loss':^14} | "
    f"{'Validation Loss':^17} |"
)

print("-" * 47)

for epoch in range(
    int(env.EPOCHS)
):

    train_loss = trainer.train_epoch(
        train_loader
    )
    val_loss = trainer.validate_epoch(
        validation_loader
    )

    print(
            f"| {epoch + 1:^7} | "
            f"{train_loss:^14.4f} | "
            f"{val_loss:^17.4f} |"
        )


    if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
    
            save_model(
                path=f"{env.SAVED_MODEL_PATH}",
                model=model,
            )
            print(
                f"  ✓ New best model saved "
                f"(epoch={best_epoch}, "
                f"val_loss={best_val_loss:.4f})"
            )
print()
print("****** Training Done ******")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Loss: {best_val_loss:.4f}")