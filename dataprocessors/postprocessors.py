import torch

from dataprocessors.encoders import EntityEncoder, IntentEncoder
from utils.schemas import EntityPrediction, EntityTokenPrediction, IntentPrediction


class PostProcessingLogitMapper:

    def __init__(
            self, 
            intent_encoder:IntentEncoder, 
            entity_encoder:EntityEncoder,
            intent_threshold = 0.5):

        self.intent_encoder = intent_encoder
        self.entity_encoder = entity_encoder
        self.intent_threshold = intent_threshold

    def map_intents(self, logits: torch.Tensor) -> list[IntentPrediction]:
        probabilities = torch.sigmoid(logits[0])
        predictions:IntentPrediction = []

        for idx, prob in enumerate(probabilities):
            confidence = float(prob)
            if confidence >= self.intent_threshold:
                predictions.append(IntentPrediction(
                    prediction_id=idx,
                    label=self.intent_encoder.id_to_label[idx],
                    confidence=confidence
                ))

        return predictions

    def map_entities(
            self, 
            logits: torch.Tensor,
            text:str,
            offset_mapping:torch.Tensor) -> list[EntityTokenPrediction]:

        token_predictions = torch.argmax(logits, dim=-1)[0]

        enitites:list[EntityTokenPrediction] = []

        for token_id, offset in zip(token_predictions, offset_mapping):
            label_id = int(token_id)
            label = self.entity_encoder.id_to_label[label_id]
            confidence = float(torch.softmax(logits[0], dim=-1)[0][label_id])

            if label == "O":
                continue

            start_index = int(offset[0])
            end_index = int(offset[1])

            if start_index == end_index:
                continue

            enitites.append(EntityTokenPrediction(
                prediction_id=label_id,
                label=label,
                confidence=confidence,
                start_index=start_index,
                end_index=end_index,
                value=text[start_index:end_index]
            ))

        return enitites


class EntityCombiner:

    def combine(
        self,
        predictions:list[EntityTokenPrediction]
    ) -> list[EntityPrediction]:

        entities:list[EntityPrediction] = []
        current:EntityPrediction | None = None


        for prediction in predictions:

            label = prediction.label


            if label == "O":
                if current is not None:
                    entities.append(current)
                    current = None

                continue


            if label.startswith("B-"):

                if current is not None:
                    entities.append(current)


                current = EntityPrediction(
                    prediction_id=prediction.prediction_id,
                    label=label[2:],
                    confidence=prediction.confidence,
                    start_index=prediction.start_index,
                    end_index=prediction.end_index,
                    value=prediction.value
                )

                continue


            if label.startswith("I-"):

                entity_label = label[2:]


                if current is None:
                    current = EntityPrediction(
                        prediction_id=prediction.prediction_id,
                        label=entity_label,
                        confidence=prediction.confidence,
                        start_index=prediction.start_index,
                        end_index=prediction.end_index,
                        value=prediction.value,
                    )

                    continue


                if current.label != entity_label:

                    entities.append(current)

                    current = EntityPrediction(
                        prediction_id=prediction.prediction_id,
                        label=entity_label,
                        confidence=prediction.confidence,
                        start_index=prediction.start_index,
                        end_index=prediction.end_index,
                        value=prediction.value,
                    )

                    continue


                current.end_index = prediction.end_index
                current.value += prediction.value

                current.confidence = max(
                    current.confidence,
                    prediction.confidence
                )


        # IMPORTANT: after loop finishes
        if current is not None:
            entities.append(current)


        return entities
