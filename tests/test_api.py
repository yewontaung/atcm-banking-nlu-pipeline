import torch
from transformers import AutoTokenizer

from dataprocessors.encoders.bio import LabelBIOEncoder
from dataprocessors.postprocessors.prediction_builder import PredictionBuilder
from inference.mappers.model_02 import Model02PredictionMapper
from inference.predictor import Predictor
from models.model_02_token_intent_transformer_model.model import Model02BankingNLUTransformerModel
from utils import env
from utils.loader import load_modelname, load_saved_model


predictor:Predictor = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_predictor():
    global predictor
    intent_encoder = LabelBIOEncoder.from_file(env.INTENT_META_FILE)
    entity_encoder = LabelBIOEncoder.from_file(env.ENTITY_META_FILE)
    model = Model02BankingNLUTransformerModel(
        model_name="xlm-roberta-base",
        intent_count=intent_encoder.no_of_labels,
        entity_count=entity_encoder.no_of_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    mapper = Model02PredictionMapper(
        intent_encoder=intent_encoder,
        entity_encoder=entity_encoder
    )
    builder = PredictionBuilder()

    saved_model = load_saved_model(model, saved_model_path=f"{env.SAVED_MODEL_PATH}/{load_modelname()}", device=DEVICE)
    predictor = Predictor(
        model=saved_model,
        device=DEVICE,
        tokenizer=tokenizer,
        mapper=mapper,
        builder=builder,
    )

def main():
    load_predictor()
    loop = True
    if predictor is None:
        loop = False
    while loop:
        prompt = input("Enter prompt : ")
        if prompt.lower() == "exit" or prompt.lower() == "0":
            break
        prediction = predictor.predict(prompt)
        print("======== Model Result =========")
        print(prediction.model_dump_json(indent=2))
        print("======== ************ =========")

    print("===== GOOD BYE =====")

if __name__ == "__main__":
    main()
