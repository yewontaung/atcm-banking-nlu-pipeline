import torch

from transformers import AutoTokenizer

from dataprocessors.processors import PostProcessingLogitMapper
from models.transformer_model import (
    BankingNLUTransformerModel
)

from dataprocessors.encoders import (
    IntentEncoder,
    EntityEncoder
)
from utils import env
from utils.schemas import ModelOutput

intent_encoder = IntentEncoder.from_file("./metadata/intents.json")
entity_encoder = EntityEncoder.from_file("./metadata/entities.json")

model = BankingNLUTransformerModel(
    model_name="xlm-roberta-base",
    intent_count=intent_encoder.no_of_lables,
    entity_count=entity_encoder.no_of_entities
)

checkpoint = torch.load(
    f"{env.CHECKPOINT_PATH}/banking_nlu.pt",
    map_location="cpu"
)

model.load_state_dict(checkpoint["model"])

model.eval()

text = "မင်းမောင်က ၃၀၀၀ ကျပ် ပေးပို့ထားတာကို စစ်ဆေးပါ။"
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

encoded = tokenizer(
    text,
    return_tensors="pt",
    return_offsets_mapping=True,
    padding=True,
    truncation=True
)

threshold = 0.5

mapper = PostProcessingLogitMapper(
    intent_encoder=intent_encoder,
    entity_encoder=entity_encoder,
    intent_threshold=0.1
)

with torch.no_grad():
    output:ModelOutput = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    intents = mapper.map_intents(output.intent_logits)
    print(intents)