import streamlit as st

from banking_nlu.dataprocessors.postprocessors.prediction_builder import PredictionBuilder
from banking_nlu.inference.mappers.model_02 import Model02PredictionMapper

@st.cache_resource
def load_predictor():
    import torch
    from transformers import AutoTokenizer

    from banking_nlu.dataprocessors.encoders.bio import LabelBIOEncoder
    from banking_nlu.inference.predictor import Predictor
    from banking_nlu.models.model_02_token_intent_transformer_model.model import Model02BankingNLUTransformerModel
    from banking_nlu.utils import env
    from banking_nlu.utils.loader import load_modelname, load_saved_model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    intent_encoder = LabelBIOEncoder.from_file(env.INTENT_META_FILE)
    entity_encoder = LabelBIOEncoder.from_file(env.ENTITY_META_FILE)
    model = Model02BankingNLUTransformerModel(
        model_name="xlm-roberta-base",
        intent_count=intent_encoder.no_of_labels,
        entity_count=entity_encoder.no_of_labels
    )

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    mapper = Model02PredictionMapper(intent_encoder, entity_encoder)
    builder = PredictionBuilder()

    saved_model = load_saved_model(model, f"{env.SAVED_MODEL_PATH}/{load_modelname()}", DEVICE)
    predictor = Predictor(
        model=saved_model,
        tokenizer=tokenizer,
        device=DEVICE,
        mapper=mapper,
        builder=builder,
    )
    return predictor