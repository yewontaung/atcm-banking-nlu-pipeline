import argparse

import torch
from transformers import AutoTokenizer

from dataprocessors.encoders.bio import LabelBIOEncoder
from dataprocessors.postprocessors.prediction_builder import PredictionBuilder
from inference.mappers.model_02 import Model02PredictionMapper
from inference.predictor import Predictor
from models.model_02_token_intent_transformer_model.model import Model02BankingNLUTokenIntentTransformerModel
from utils import env
from utils.loader import load_saved_model, load_modelname


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_predictor() -> Predictor:
    intent_encoder = LabelBIOEncoder.from_file("./metadata/intents.json")
    entity_encoder = LabelBIOEncoder.from_file("./metadata/entities.json")

    model = Model02BankingNLUTokenIntentTransformerModel(
        model_name="xlm-roberta-base",
        intent_count=intent_encoder.no_of_labels,
        entity_count=entity_encoder.no_of_labels
    )

    checkpoint_path = f"{env.SAVED_MODEL_PATH}/{load_modelname()}"

    model = load_saved_model(model, checkpoint_path, DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    mapper = Model02PredictionMapper(
        intent_encoder, entity_encoder
    )
    builder = PredictionBuilder()

    return Predictor(
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