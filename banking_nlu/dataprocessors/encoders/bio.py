import json
from typing import Final


class BIOLabelEncoder:

    def __init__(self, items:list[str]):
        labels = ["O"]
        for item in items:
            labels.append(f"B-{item}")
            labels.append(f"I-{item}")

        self.label_to_id:dict[str, int] = {
            label:index for index, label in enumerate(labels)
        }

        self.id_to_label:dict[int, str] = {
            index:label for index, label in enumerate(labels)
        }

        self.no_of_labels:Final[int] = len(labels)

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
    def from_file(path:str) -> "BIOLabelEncoder":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return BIOLabelEncoder(items=[item["label"] for item in data])
        