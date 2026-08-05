import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from banking_nlu.dataloader.collator import NLUCollator
from banking_nlu.dataloader.dataset import NLUDataset
from banking_nlu.dataprocessors.encoders.bio import LabelBIOEncoder
from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.dataprocessors.tokenizers import Model02TokenizationProcessor, TextTokenizer
from banking_nlu.models.model_02_token_intent_transformer_model.loss import Model02TokenIntentEntityLoss
from banking_nlu.models.model_02_token_intent_transformer_model.model import Model02BankingNLUTransformerModel
from train.trainer import NLUModelTrainer
from banking_nlu.utils import env
from banking_nlu.utils.checkpoint import save_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

intent_encoder = LabelBIOEncoder.from_file("./metadata/intents.json")
entity_encoder = LabelBIOEncoder.from_file("./metadata/entities.json")

processor = DataPreProcessor()

processed = processor.process_file(f"{env.TRAINING_FILE}")

tokenizer = TextTokenizer(AutoTokenizer.from_pretrained("xlm-roberta-base"))

tokenization_processor = Model02TokenizationProcessor(
    tokenizer=tokenizer,
    intent_encoder=intent_encoder,
    entity_encoder=entity_encoder
)

tokenized = [tokenization_processor.process(item) for item in processed]

collator = NLUCollator(
    pad_token_id=tokenizer.tokenizer.pad_token_id,
    intent_mode="token_span"
)

dataset = NLUDataset(tokenized)

loader = DataLoader(
    dataset=dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collator
)

model = Model02BankingNLUTransformerModel(
    model_name="xlm-roberta-base",
    intent_count=intent_encoder.no_of_labels,
    entity_count=entity_encoder.no_of_labels,
)

model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-5)

loss_fn = Model02TokenIntentEntityLoss()

trainer = NLUModelTrainer(
    model=model,
    device=DEVICE,
    optimizer=optimizer,
    loss_fn=loss_fn
)

for epoch in range(int(env.EPOCHS)):
    loss = trainer.train_epoch(loader)

    print(f"Epoch {epoch + 1}: {loss}")

save_model(
    path=f"{env.SAVED_MODEL_PATH}",
    model=model,
)

print("****** Training Done *******")