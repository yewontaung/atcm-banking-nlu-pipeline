import torch

from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.dataprocessors.tokenizers import Model02TokenizationProcessor, TextTokenizer
from banking_nlu.factory.model_02_utils import Model02EvaluatorFactory
from banking_nlu.utils import env


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    evaluator = Model02EvaluatorFactory.from_default(DEVICE)
    preprocessor = DataPreProcessor()
    processed = preprocessor.process_file(env.TESTING_FILE)
    tokenization_processor = Model02TokenizationProcessor(
        tokenizer=TextTokenizer(evaluator.tokenizer),
        intent_encoder=evaluator.intent_encoder,
        entity_encoder=evaluator.entity_encoder,
    )
    tokenized = [tokenization_processor.process(item) for item in processed]
    result = evaluator.get_evaluation_data(tokenized=tokenized)
    print(result.intent_truth.grounded_truth)
    print(result.intent_truth.predicted_truth)

    print("======================================")
    print(result.entity_truth.grounded_truth)
    print(result.entity_truth.grounded_truth)


if __name__ == "__main__":
    main()