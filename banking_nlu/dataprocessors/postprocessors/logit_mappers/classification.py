import torch

from banking_nlu.dataprocessors.encoders.bio import BIOLabelEncoder
from banking_nlu.dataprocessors.encoders.classification import IntentClassificationEncoder
from banking_nlu.utils.schemas import TokenPrediction, ClassifiedIntentPrediction


class ClassificationLogitMapper:

    def __init__(
            self, 
            intent_encoder:IntentClassificationEncoder, 
            entity_encoder:BIOLabelEncoder,
            intent_threshold = 0.5):

        self.intent_encoder = intent_encoder
        self.entity_encoder = entity_encoder
        self.intent_threshold = intent_threshold

    def map_intents(self, logits: torch.Tensor) -> list[ClassifiedIntentPrediction]:
        probabilities = torch.sigmoid(logits[0])
        predictions:ClassifiedIntentPrediction = []

        for idx, prob in enumerate(probabilities):
            confidence = float(prob)
            if confidence >= self.intent_threshold:
                predictions.append(ClassifiedIntentPrediction(
                    prediction_id=idx,
                    label=self.intent_encoder.id_to_label[idx],
                    confidence=confidence
                ))

        return predictions

    def map_entities(
            self, 
            logits: torch.Tensor,
            text:str,
            offset_mapping:torch.Tensor) -> list[TokenPrediction]:

        token_predictions = torch.argmax(logits, dim=-1)[0]

        enitites:list[TokenPrediction] = []

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

            enitites.append(TokenPrediction(
                prediction_id=label_id,
                label=label,
                confidence=confidence,
                start_index=start_index,
                end_index=end_index,
                value=text[start_index:end_index]
            ))

        return enitites
