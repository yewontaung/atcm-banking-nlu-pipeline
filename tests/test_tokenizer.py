from dataprocessors.preprocessors import DataPreProcessor

from dataprocessors.tokenizers import TokenizationProcessor

from dataprocessors.encoders import (
    IntentEncoder,
    EntityEncoder
)

intent_encoder = IntentEncoder.from_file("./metadata/intents.json")

entity_encoder = EntityEncoder.from_file("./metadata/entities.json")

processor = DataPreProcessor()

samples = processor.process_file(
    "datasets/testing.json"
)

tokenizer = TokenizationProcessor(
    "xlm-roberta-base",
    intent_encoder,
    entity_encoder
)

result = tokenizer.process(
    samples[0]
)

print(result)