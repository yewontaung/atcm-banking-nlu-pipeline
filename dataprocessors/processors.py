import json

import torch

from dataprocessors.encoders import EntityEncoder, IntentEncoder
from dataprocessors.validators import DataValidator
from utils.schemas import EntityPrediction, EntitySpan, ExportedDataset, IntentPrediction, IntentSpan, ProcessedDataset


class DataPreProcessor:

    def __init__(self):
        self.validator = DataValidator()

    def load_file(self, path:str) -> list[ExportedDataset]:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return [ExportedDataset(**item) for item in data]

    def process_file(self, path:str) -> list[ProcessedDataset]:
        dataset = self.load_file(path)
        return [self.process_item(item) for item in dataset]

    def process_item(self, item:ExportedDataset) -> ProcessedDataset:
        self.validator.validate(item)
        processed_intents:list[IntentSpan] = []
        for intent in item.intents:
            entities = [
                EntitySpan(
                    label=entity.label,
                    start_index=entity.start_index,
                    end_index=entity.end_index
                ) for entity in intent.entities
            ]

            processed_intents.append(IntentSpan(
                label=intent.label,
                start_index=intent.start_index,
                end_index=intent.end_index,
                entities=entities
            ))

        return ProcessedDataset(
            text=item.text,
            intents=processed_intents
        )

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
            offset_mapping:torch.Tensor) -> list[EntityPrediction]:

        token_predictions = torch.argmax(logits, dim=-1)[0]

        enitites:list[EntityPrediction] = []

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

            enitites.append(EntityPrediction(
                prediction_id=label_id,
                label=label,
                confidence=confidence,
                start_index=start_index,
                end_index=end_index,
                value=text[start_index:end_index]
            ))

        return enitites
