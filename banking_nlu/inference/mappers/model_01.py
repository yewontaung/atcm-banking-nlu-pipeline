from banking_nlu.inference.mappers.base import BasePredictionMapper

from banking_nlu.dataprocessors.postprocessors.logit_mappers.classification import (
    ClassificationLogitMapper
)

from banking_nlu.dataprocessors.postprocessors.logit_mappers.bio import (
    BIOCombiner
)


class Model01PredictionMapper(
    BasePredictionMapper
):

    def __init__(
        self,
        intent_encoder,
        entity_encoder,
        threshold
    ):

        self.logit_mapper = ClassificationLogitMapper(
            intent_encoder=intent_encoder,
            entity_encoder=entity_encoder,
            intent_threshold=threshold
        )

        self.entity_combiner = BIOCombiner()



    def map(
        self,
        text,
        outputs,
        offset_mapping
    ):

        intents = self.logit_mapper.map_intents(
            outputs.intent_logits
        )


        entity_tokens = self.logit_mapper.map_entities(
            outputs.entity_logits,
            text,
            offset_mapping
        )


        entities = self.entity_combiner.combine(
            text,
            entity_tokens
        )


        return {
            "text":text,
            "intents":intents,
            "entities":entities
        }