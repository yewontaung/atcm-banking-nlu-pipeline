import torch

from banking_nlu.dataprocessors.encoders.bio import LabelBIOEncoder
from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.dataprocessors.tokenizers import Model02TokenizationProcessor, TextTokenizer
from banking_nlu.factory.model_02_utils import Model02EvaluatorFactory
from banking_nlu.utils import env

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

intent_encoder = LabelBIOEncoder.from_file(env.INTENT_META_FILE)
entity_encoder = LabelBIOEncoder.from_file(env.ENTITY_META_FILE)

processor = DataPreProcessor()
processed = processor.process_file(env.TESTING_FILE)
evaluator = Model02EvaluatorFactory.from_default(DEVICE)
tokenization_processor = Model02TokenizationProcessor(
    tokenizer=TextTokenizer(evaluator.tokenizer),
    intent_encoder=evaluator.intent_encoder,
    entity_encoder=evaluator.entity_encoder
)
tokenized = [tokenization_processor(item) for item in processed]

def evaluate():
    result = evaluator.get_evaluation_data(tokenized)
    evaluation = evaluator.calculate_evaluation_result(result)

    print(evaluation.model_dump_json(indent=2))

if __name__ == "__main__":
    evaluate()