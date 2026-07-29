import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


from dataprocessors.encoders.classification import (
    IntentClassificationEncoder
)

from dataprocessors.encoders.bio import (
    LabelBIOEncoder
)


from dataprocessors.preprocessors import (
    DataPreProcessor
)


from dataprocessors.tokenizers import (
    TextTokenizer,
    TransformerModelTokenizationProcessor
)


from dataloader.dataset import (
    NLUDataset
)


from dataloader.collator import (
    NLUCollator
)


from models.model_01_transformer_model.model import (
    BankingNLUTransformerModel
)


from models.model_01_transformer_model.loss import (
    TransformerNLULoss
)


from train.trainer import NLUModelTrainer

from utils import env

from utils.checkpoint import save_checkpoint
from utils.loader import load_modelname



DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# ======================
# Encoders
# ======================

intent_encoder = IntentClassificationEncoder.from_file(
    "./metadata/intents.json"
)

entity_encoder = LabelBIOEncoder.from_file(
    "./metadata/entities.json"
)



# ======================
# Data
# ======================

processor = DataPreProcessor()

processed = processor.process_file(
    env.TRAINING_FILE
)



tokenizer = TextTokenizer(
    "xlm-roberta-base"
)


tokenizer_processor = (
    TransformerModelTokenizationProcessor(
        tokenizer,
        intent_encoder,
        entity_encoder
    )
)


tokenized = [
    tokenizer_processor.process(item)
    for item in processed
]


dataset = NLUDataset(tokenized)



collator = NLUCollator(
    pad_token_id=tokenizer.tokenizer.pad_token_id,
    intent_mode="classification"
)



loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
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



for epoch in range(
    int(env.EPOCHS)
):

    loss = trainer.train_epoch(
        loader
    )


    print(
        f"Epoch {epoch+1}: {loss}"
    )



# ======================
# Save
# ======================

save_checkpoint(
    path=f"{env.SAVED_MODEL_PATH}/{load_modelname()}",
    model=model,
    optimizer=optimizer,
    metadata={
        "intent_encoder":
            intent_encoder.label_to_id,

        "entity_encoder":
            entity_encoder.label_to_id
    }
)

print("Training Done.")