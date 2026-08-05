import argparse

import torch
from transformers import AutoTokenizer

from banking_nlu.dataprocessors.encoders.bio import BIOLabelEncoder
from banking_nlu.dataprocessors.postprocessors.prediction_builder import PredictionBuilder
from banking_nlu.inference.mappers.model_02 import Model02PredictionMapper
from banking_nlu.inference.predictor import BankingNLUPredictor
from banking_nlu.models.model_02_token_intent_transformer_model.model import Model02BankingNLUTransformerModel
from banking_nlu.utils import env
from banking_nlu.utils.loader import load_saved_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_predictor() -> BankingNLUPredictor:
    intent_encoder = BIOLabelEncoder.from_file("./metadata/intents.json")
    entity_encoder = BIOLabelEncoder.from_file("./metadata/entities.json")

    model = Model02BankingNLUTransformerModel(
        model_name="xlm-roberta-base",
        intent_count=intent_encoder.no_of_labels,
        entity_count=entity_encoder.no_of_labels
    )

    checkpoint_path = f"{env.SAVED_MODEL_PATH}"

    model = load_saved_model(model, checkpoint_path, DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    mapper = Model02PredictionMapper(
        intent_encoder, entity_encoder
    )
    builder = PredictionBuilder()

    return BankingNLUPredictor(
        model, tokenizer, mapper, DEVICE, builder
    )

def parse_args():

    parser = argparse.ArgumentParser(
        description="Model 2 NLU inference"
    )


    parser.add_argument(
        "-m",
        "--message",
        type=str,
        required=False,
        default=env.TEST_PROMPT,
        help="Input Burmese sentence"
    )


    return parser.parse_args()

def main():
    args = parse_args()
    predictor = create_predictor()
    prediction = predictor.predict(
        args.message
    )
    print("========= Prediction =========")
    print(prediction.model_dump_json())

if __name__ == "__main__":
    main()