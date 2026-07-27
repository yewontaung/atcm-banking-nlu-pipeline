import argparse

import torch
from transformers import AutoTokenizer

from dataprocessors.encoders.bio import LabelBIOEncoder
from dataprocessors.postprocessors.prediction_builder import PredictionBuilder
from inference.mappers.model_02 import Model02PredictionMapper
from inference.predictor import Predictor
from models.model_02_token_intent_transformer_model.model import BankingNLUTokenIntentTransformerModel
from utils import env
from utils.loader import load_checkpoint, load_modelname


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_predictor() -> Predictor:
    intent_encoder = LabelBIOEncoder.from_file("./metadata/intents.json")
    entity_encoder = LabelBIOEncoder.from_file("./metadata/entities.json")

    model = BankingNLUTokenIntentTransformerModel(
        model_name="xlm-roberta-base",
        intent_count=intent_encoder.no_of_labels,
        entity_count=entity_encoder.no_of_labels
    )

    checkpoint_path = f"{env.CHECKPOINT_PATH}/{load_modelname()}"

    model = load_checkpoint(model, checkpoint_path, DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    mapper = Model02PredictionMapper(
        intent_encoder, entity_encoder
    )

    return Predictor(
        model, tokenizer, mapper, DEVICE
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
    result = predictor.predict(
        args.message
    )
    builder = PredictionBuilder()

    print(
        f"\nTEXT : {result['text']}"
    )
    print(
        "\n===== Intents ====="
    )
    for intent in result["intents"]:
        print(intent)
    print(
        "\n===== Entities ====="
    )
    for entity in result["entities"]:
        print(entity)

    intents = result["intents"]
    entities = result["entities"]
    prediction = builder.build(args.message, intents, entities)

    print(prediction)

if __name__ == "__main__":
    main()