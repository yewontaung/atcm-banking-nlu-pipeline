from banking_nlu.dataprocessors.encoders.bio import BIOLabelEncoder
from banking_nlu.evaluations.evaluators import Model02Evaluator
from banking_nlu.inference.mappers.model_02 import Model02PredictionMapper
from banking_nlu.models.model_02_token_intent_transformer_model.model import Model02BankingNLUTransformerModel
from banking_nlu.utils import env
from banking_nlu.utils.loader import load_saved_model


class Model02EvaluatorFactory:

    @staticmethod
    def from_default(device:str):

        intent_encoder = BIOLabelEncoder.from_file(env.INTENT_META_FILE)
        entity_encoder = BIOLabelEncoder.from_file(env.ENTITY_META_FILE)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        mapper = Model02PredictionMapper(intent_encoder, entity_encoder)

        model = Model02BankingNLUTransformerModel(
            model_name="xlm-roberta-base",
            intent_count=intent_encoder.no_of_labels,
            entity_count=entity_encoder.no_of_labels
        )

        saved_model = load_saved_model(model, f"{env.SAVED_MODEL_PATH}", device)

        return Model02Evaluator(
            model=saved_model,
            tokenizer=tokenizer,
            intent_encoder=intent_encoder,
            entity_encoder=entity_encoder,
            mapper=mapper,
            device=device,
        )

