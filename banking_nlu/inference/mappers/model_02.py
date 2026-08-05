from torch import Tensor

from banking_nlu.dataprocessors.encoders.bio import BIOLabelEncoder
from banking_nlu.inference.mappers.base import BasePredictionMapper

from banking_nlu.dataprocessors.postprocessors.logit_mappers.bio import (
    BIOCombiner,
    BIOLogitMapper,
)
from banking_nlu.utils.schemas import Model02LogitOutput


class Model02PredictionMapper(BasePredictionMapper):

    def __init__(
        self,
        intent_encoder:BIOLabelEncoder,
        entity_encoder:BIOLabelEncoder,
    ):

        self.intent_mapper = BIOLogitMapper(intent_encoder)
        self.entity_mapper = BIOLogitMapper(entity_encoder)
        self.combiner = BIOCombiner()


    def map(self, text:str, outputs:Model02LogitOutput, offset_mapping:Tensor):
        intent_tokens = self.intent_mapper.map(
            outputs.intent_logits,
            text,
            offset_mapping
        )

        entity_tokens = self.entity_mapper.map(
            outputs.entity_logits,
            text,
            offset_mapping
        )

        intents = self.combiner.combine(
            text,
            intent_tokens
        )

        entity_spans = self.combiner.combine(
            text,
            entity_tokens
        )

        return {
            "text":text,
            "intents":intents,
            "entities":entity_spans
        }