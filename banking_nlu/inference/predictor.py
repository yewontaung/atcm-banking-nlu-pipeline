import torch

from banking_nlu.dataprocessors.encoders.bio import BIOLabelEncoder
from banking_nlu.dataprocessors.postprocessors.prediction_builder import PredictionBuilder
from banking_nlu.inference.mappers.model_02 import Model02PredictionMapper
from banking_nlu.models.model_02_token_intent_transformer_model.model import Model02BankingNLUTransformerModel
from banking_nlu.utils.loader import load_saved_model
from banking_nlu.utils.schemas import ModelPrediction


class BankingNLUPredictor:

    def __init__(
        self,
        model,
        tokenizer,
        mapper,
        device,
        builder:PredictionBuilder
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mapper = mapper
        self.device = device
        self.builder = builder


    def predict(self, text:str) -> ModelPrediction:

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True
        )


        input_ids = encoded["input_ids"].to(
            self.device
        )

        attention_mask = encoded["attention_mask"].to(
            self.device
        )


        self.model.eval()

        with torch.no_grad():

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )


        result = self.mapper.map(
            text=text,
            outputs=outputs,
            offset_mapping=encoded["offset_mapping"][0]
        )

        return self.builder.build(
            text=result["text"],
            intents=result["intents"],
            entities=result["entities"]
        )

    """
    Model 02 is exposed for use. 
    """
    @staticmethod
    def load(
        model_name:str, 
        saved_model_path:str, 
        intent_metadata_path:str,
        entity_metadata_path:str, 
        device:str) -> "BankingNLUPredictor":

        intent_encoder = BIOLabelEncoder.from_file(intent_metadata_path)
        entity_encoder = BIOLabelEncoder.from_file(entity_metadata_path)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        builder = PredictionBuilder()

        model = Model02BankingNLUTransformerModel(
            model_name=model_name,
            intent_count=intent_encoder.no_of_labels,
            entity_count=entity_encoder.no_of_labels,)

        mapper =  Model02PredictionMapper(
            intent_encoder=intent_encoder,
            entity_encoder=entity_encoder
        )
        saved_model = load_saved_model(model, saved_model_path, device)
        return BankingNLUPredictor(
            model=saved_model,
            tokenizer=tokenizer,
            mapper=mapper,
            builder=builder,
            device=device,
        )