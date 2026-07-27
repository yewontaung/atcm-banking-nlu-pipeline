from dataprocessors.encoders.bio import LabelBIOEncoder
from dataprocessors.preprocessors import DataPreProcessor

from dataprocessors.tokenizers import TextTokenizer, TransformerModelTokenizationProcessor, SpanBasedModelTokenizationProcessor

from dataprocessors.encoders.classification import IntentClassificationEncoder

intent_encoder = LabelBIOEncoder.from_file("./metadata/intents.json")

entity_encoder = LabelBIOEncoder.from_file("./metadata/entities.json")

processor = DataPreProcessor()

samples = processor.process_file(
    "datasets/training.json"
)
tokenizer = TextTokenizer("xlm-roberta-base")
tokenization_processor = SpanBasedModelTokenizationProcessor(
    tokenizer,
    intent_encoder,
    entity_encoder
)

result = tokenization_processor.process(
    samples[0]
)

print(result)