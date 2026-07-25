import json

import torch

from dataprocessors.encoders import EntityEncoder, IntentEncoder
from dataprocessors.validators import DataValidator
from utils.schemas import EntityTokenPrediction, EntitySpan, ExportedDataset, IntentPrediction, IntentSpan, ProcessedDataset


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
