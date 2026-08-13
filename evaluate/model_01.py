import torch
from transformers import AutoTokenizer

from banking_nlu.dataprocessors.encoders.bio import BIOLabelEncoder
from banking_nlu.dataprocessors.encoders.classification import IntentClassificationEncoder
from banking_nlu.dataprocessors.postprocessors.logit_mappers.classification import ClassificationLogitMapper
from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.dataprocessors.tokenizers import Model01TransformerModelTokenizationProcessor, TextTokenizer
from banking_nlu.evaluations.matrices import calculate_evaluation_result
from banking_nlu.evaluations.models.model_01_evaluator import Model01Evaluator
from banking_nlu.models.model_01_transformer_model.model import Model01BankingNLUTransformerModel
from banking_nlu.utils import env
from banking_nlu.utils.loader import load_saved_model


def evaluate():

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    intent_encoder = IntentClassificationEncoder.from_file(env.INTENT_META_FILE)
    entity_encoder = BIOLabelEncoder.from_file(env.ENTITY_META_FILE)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model01BankingNLUTransformerModel(
        model_name="xlm-roberta-base",
        intent_count=intent_encoder.no_of_lables,
        entity_count=entity_encoder.no_of_labels,
    )

    saved_model = load_saved_model(model, env.SAVED_MODEL_PATH, DEVICE)


    tokenization_processor = Model01TransformerModelTokenizationProcessor(
        tokenizer=TextTokenizer(tokenizer),
        intent_encoder=intent_encoder,
        entity_encoder=entity_encoder,
    )

    mapper = ClassificationLogitMapper(
        intent_encoder=intent_encoder,
        entity_encoder=entity_encoder
    )


    evaluator = Model01Evaluator(
        model=saved_model,
        tokenizer=tokenizer,
        intent_encoder=intent_encoder,
        entity_encoder=entity_encoder,
        mapper=mapper,
        device=DEVICE,
    )
    preprocessor = DataPreProcessor()
    processed = preprocessor.process_file(env.TESTING_FILE)
    tokenized = [tokenization_processor.process(item) for item in processed]

    evaluation_data = evaluator.get_evaluation_data(tokenized)

    print(evaluation_data.model_dump_json())

    result = evaluator.calculate_evaluation_result()
    print(f"Accuracy : {result.accuracy}")
    print(f"F1       : {result.f1}")
    with open(env.EVALUATION_FILE, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

if __name__ == "__main__":
    evaluate()