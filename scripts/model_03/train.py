import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from banking_nlu.dataloader.collator import NLUCollator
from banking_nlu.dataloader.dataset import NLUDataset
from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.dataprocessors.tokenizers import (
    Model03TokenizationProcessor,
    TextTokenizer,
)
from banking_nlu.models.model_03_llm_model.model import (
    Model03LLMBasedClassificationModel,
)
from scripts.trainer import NLUModelTrainer
from banking_nlu.utils import env
from banking_nlu.utils.checkpoint import save_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


# ============================================================
# Data Processing
# ============================================================

processor = DataPreProcessor()

train_processed = processor.process_file(
    env.TRAIN_JSON
)

validation_processed = processor.process_file(
    env.VALIDATE_JSON
)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = TextTokenizer(
    AutoTokenizer.from_pretrained(
        MODEL_NAME
    )
)

tokenization_processor = Model03TokenizationProcessor(
    tokenizer=tokenizer
)


train_tokenized = [
    tokenization_processor.process(item)
    for item in train_processed
]

validation_tokenized = [
    tokenization_processor.process(item)
    for item in validation_processed
]


# ============================================================
# Dataset
# ============================================================

train_dataset = NLUDataset(train_tokenized)

validation_dataset = NLUDataset(
    validation_tokenized
)


# ============================================================
# Collator
# ============================================================

collator = NLUCollator(
    pad_token_id=tokenizer.tokenizer.pad_token_id,
)


# ============================================================
# DataLoader
# ============================================================

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collator,
)

val_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collator,
)


# ============================================================
# Model
# ============================================================

model = Model03LLMBasedClassificationModel(
    model_name=MODEL_NAME,
    tokenizer=tokenizer.tokenizer,
    adapter_path=None,
    device=DEVICE,
)

model.to(DEVICE)


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=2e-4,
)


# ============================================================
# Trainer
# ============================================================

trainer = NLUModelTrainer(
    model=model,
    device=DEVICE,
    optimizer=optimizer,
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")
best_epoch = 0

print()

print(
    f"| {'Epoch':^7} | "
    f"{'Train Loss':^14} | "
    f"{'Validation Loss':^17} |"
)

print("-" * 47)


for epoch in range(int(env.EPOCHS)):

    train_loss = trainer.train_epoch(
        train_loader
    )

    val_loss = trainer.validate_epoch(
        val_loader
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