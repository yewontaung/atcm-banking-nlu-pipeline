import json
from typing import Final


class IntentEncoder:

    def __init__(self, labels:list[str]):
        self.label_to_id:dict[str, int] = {
            label:index for index, label in enumerate(labels)
        }

        self.id_to_label:dict[int, str] = {
            index:label for index, label in enumerate(labels)
        }

        self.no_of_lables:Final[int] = len(labels)

    def encode(self, labels:list[str]) -> list[int]:
        result = [0 for _ in range(self.no_of_lables)]

        for label in labels:
            if label not in self.label_to_id:
                raise ValueError(f"Unknown intent label: {label}")

            index = self.label_to_id[label]
            result[index] = 1
        return result

    def decode(self, ids:list[str]) -> list[str]:
        return [
            self.id_to_label[index]
            for index, value in enumerate(ids)
            if value == 1
        ]

    @staticmethod
    def from_file(path:str) -> "IntentEncoder":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return IntentEncoder(labels=[item["label"] for item in data])

class EntityEncoder:

    def __init__(self, entities:list[str]):
        labels = ["O"]
        for entity in entities:
            labels.append(f"B-{entity}")
            labels.append(f"I-{entity}")

        self.label_to_id:dict[str, int] = {
            label:index for index, label in enumerate(labels)
        }

        self.id_to_label:dict[int, str] = {
            index:label for index, label in enumerate(labels)
        }

        self.no_of_entities:Final[int] = len(labels)

    def encode(self, entities:list[str]) -> list[int]:
        return [
            self.label_to_id[label]
            for label in entities
        ]

    def decode(self, ids:list[int]) -> list[str]:
        return [
            self.id_to_label[id_] for id_ in ids
        ]

    @staticmethod
    def from_file(path:str) -> "EntityEncoder":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return EntityEncoder(entities=[item["label"] for item in data])
        