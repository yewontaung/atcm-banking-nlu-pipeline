from torch import Tensor

from dataprocessors.encoders.bio import LabelBIOEncoder
from inference.mappers.base import BasePredictionMapper

from dataprocessors.postprocessors.logit_mappers.bio import (
    BIOCombiner,
    BIOLogitMapper,
)
from utils.schemas import TokenIntentModelOutput


class Model02PredictionMapper(BasePredictionMapper):

    def __init__(
        self,
        intent_encoder:LabelBIOEncoder,
        entity_encoder:LabelBIOEncoder,
    ):

        self.intent_mapper = BIOLogitMapper(intent_encoder)
        self.entity_mapper = BIOLogitMapper(entity_encoder)
        self.combiner = BIOCombiner()


    def map(self, text:str, outputs:TokenIntentModelOutput, offset_mapping:Tensor):
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