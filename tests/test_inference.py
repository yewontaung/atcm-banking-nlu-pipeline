import torch

from transformers import AutoTokenizer

from dataprocessors.postprocessors import EntityCombiner, PostProcessingLogitMapper
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
    f"{env.CHECKPOINT_PATH}/{env.SAVED_MODEL_NAME_PREFIX}_{env.TRAINING_DATASIZE}.pt",
    map_location="cpu"
)

model.load_state_dict(checkpoint["model"])

model.eval()

text = env.TEST_PROMPT
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
    intent_threshold=float(env.INTENT_THRESHOLD)
)

entity_combiner = EntityCombiner()

with torch.no_grad():
    output:ModelOutput = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    intents = mapper.map_intents(output.intent_logits)
    entity_tokens = mapper.map_entities(
        output.entity_logits,
        text,
        encoded["offset_mapping"][0]
    )
    entities = entity_combiner.combine(entity_tokens)
    print(f"TEXT : {text}")
    print("==== intent prediction ====")
    print(intents)
    print("==== entity prediction ====")
    print(entities)