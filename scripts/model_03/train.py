# scripts/model_03/train.py

import json
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.utils import env


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2
EPOCHS = int(env.EPOCHS)
LEARNING_RATE = 2e-4

MAX_LENGTH = 512

SAVE_PATH = f"{env.SAVED_MODEL_PATH}"


# ============================================================
# Dataset
# ============================================================

class Model03Dataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


# ============================================================
# Tokenization
# ============================================================

def create_training_sample(
    sample,
    tokenizer,
):
    """
    Convert the clean DataPreProcessor object into:

        input_ids
        attention_mask
        labels

    The model receives:

        prompt + expected JSON

    but loss is calculated ONLY on the expected JSON.
    """

    target = {
        "text": sample.text,
        "intents": [
            {
                "label": intent.label,
                "entities": [
                    {
                        "label": entity.label,
                        "value": sample.text[
                            entity.start_index:entity.end_index
                        ],
                    }
                    for entity in intent.entities
                ],
            }
            for intent in sample.intents
        ],
    }

    target_json = json.dumps(
        target,
        ensure_ascii=False,
    )

    prompt = f"""Extract the intents and entities from this banking request.

Input:
{sample.text}

Return JSON only.

Answer:
"""

    # ----------------------------------------
    # Tokenize prompt
    # ----------------------------------------

    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # ----------------------------------------
    # Tokenize expected answer
    # ----------------------------------------

    target_tokens = tokenizer(
        target_json,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    prompt_ids = prompt_tokens["input_ids"]
    target_ids = target_tokens["input_ids"]

    input_ids = prompt_ids + target_ids

    attention_mask = (
        [1] * len(input_ids)
    )

    # Don't calculate loss for prompt tokens.
    labels = (
        [-100] * len(prompt_ids)
        + target_ids
    )

    # ----------------------------------------
    # Limit total sequence length
    # ----------------------------------------

    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# Collator
# ============================================================

class Model03Collator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        max_length = max(
            len(item["input_ids"])
            for item in batch
        )

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:

            padding = (
                max_length
                - len(item["input_ids"])
            )

            input_ids.append(
                item["input_ids"]
                + [self.tokenizer.pad_token_id] * padding
            )

            attention_masks.append(
                item["attention_mask"]
                + [0] * padding
            )

            labels.append(
                item["labels"]
                + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


# ============================================================
# Load Data
# ============================================================

print("Loading dataset...")

processor = DataPreProcessor()

train_processed = processor.process_file(
    env.TRAIN_JSON
)

validation_processed = processor.process_file(
    env.VALIDATE_JSON
)

print(
    f"Training samples: {len(train_processed)}"
)

print(
    f"Validation samples: {len(validation_processed)}"
)


# ============================================================
# Tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Prepare Training Data
# ============================================================

print("Tokenizing training data...")

train_data = [
    create_training_sample(
        sample,
        tokenizer,
    )
    for sample in train_processed
]

print("Tokenizing validation data...")

validation_data = [
    create_training_sample(
        sample,
        tokenizer,
    )
    for sample in validation_processed
]


# ============================================================
# Dataset
# ============================================================

train_dataset = Model03Dataset(
    train_data
)

validation_dataset = Model03Dataset(
    validation_data
)


# ============================================================
# DataLoader
# ============================================================

collator = Model03Collator(
    tokenizer
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
)


# ============================================================
# Load LLM
# ============================================================

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,

    dtype=(
        torch.float16
        if DEVICE == "cuda"
        else torch.float32
    ),
)


# ============================================================
# LoRA
# ============================================================

print("Creating LoRA adapter...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()

model.to(DEVICE)


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Training Functions
# ============================================================

def train_epoch():

    model.train()

    total_loss = 0.0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(
            DEVICE
        )

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch["labels"].to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return (
        total_loss
        / len(train_loader)
    )


def validate():

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in validation_loader:

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "labels"
            ].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

    return (
        total_loss
        / len(validation_loader)
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


for epoch in range(EPOCHS):

    train_loss = train_epoch()

    val_loss = validate()

    print(
        f"| {epoch + 1:^7} | "
        f"{train_loss:^14.4f} | "
        f"{val_loss:^17.4f} |"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1

        print(
            f"  ✓ New best model "
            f"(epoch={best_epoch}, "
            f"val_loss={best_val_loss:.4f})"
        )

        model.save_pretrained(
            SAVE_PATH
        )


print()
print("****** Training Done ******")
print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Adapter saved to: {SAVE_PATH}"
)