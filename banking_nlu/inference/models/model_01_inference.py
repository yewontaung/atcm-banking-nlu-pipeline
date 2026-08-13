import torch

from banking_nlu.dataprocessors.encoders.bio import BIOLabelEncoder
from banking_nlu.dataprocessors.encoders.classification import IntentClassificationEncoder
from banking_nlu.dataprocessors.postprocessors.logit_mappers.bio import BIOCombiner
from banking_nlu.utils.schemas import Model01LogitOutput


class Model01Inference:

    def __init__(
            self,
            model,
            tokenizer,
            mapper,
            intent_encoder:IntentClassificationEncoder,
            entity_encoder:BIOLabelEncoder,
            DEVICE:str = "cpu" ):
        
        self.DEVICE = DEVICE
        self.intent_encoder = intent_encoder
        self.entity_encoder = entity_encoder
        self.model = model
        self.tokenizer = tokenizer
        self.mapper = mapper
        self.entity_combiner = BIOCombiner()


    def predict(self, text:str):

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True
        )


        with torch.no_grad():
            outputs:Model01LogitOutput = self.model(
                input_ids=encoded["input_ids"].to(self.DEVICE),
                attention_mask=encoded["attention_mask"].to(self.DEVICE)
            )
        return self.mapper.map(
            text,
            outputs,
            encoded["offset_mapping"][0],
        )


