from utils.schemas import ModelPrediction, PredictedEntity, PredictedIntent, SpanPrediction


class PredictionBuilder:

    def build(
        self,
        text:str,
        intents:list[SpanPrediction],
        entities:list[SpanPrediction]
    ) -> ModelPrediction:


        result = []


        for intent in intents:

            matched = []

            for entity in entities:

                if (
                    entity.start_index >= intent.start_index
                    and
                    entity.end_index <= intent.end_index
                ):
                    matched.append(
                        PredictedEntity(
                            label=entity.label,
                            value=entity.value
                        )
                    )


            result.append(
                PredictedIntent(
                    label=intent.label,
                    confidence=intent.confidence,
                    entities=matched
                )
            )


        return ModelPrediction(
            text=text,
            intents=result
        )