from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from transformers import AutoTokenizer, SentencePieceBackend

from banking_nlu.dataprocessors.encoders.bio import BIOLabelEncoder
from banking_nlu.dataprocessors.encoders.classification import IntentClassificationEncoder
from banking_nlu.utils.schemas import EntitySpan, ProcessedDataset, Span, TokenizedDataset, TransformerTokenizedDataset, SpanIntentTokenizedDataset

R = TypeVar("R", bound=TokenizedDataset)

class TextTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer:SentencePieceBackend = tokenizer

    def tokenize(self, text:str):
        return self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding=False
        )

class BaseTokenizationProcessor(ABC, Generic[R]):

    @abstractmethod
    def process(self, sample:ProcessedDataset) -> R:...

class BIOLabeler:

    def __init__(self, encoder:BIOLabelEncoder):
        self.encoder = encoder

    def create(self, items:list[Span], offsets:list[tuple[int, int]]) -> list[int]:
        labels = ["O" for _ in offsets]
        for item in items:
            first = True
            for index, (token_start, token_end) in enumerate(offsets):
                if token_start == token_end:
                    continue
                if token_start >= item.start_index and token_end <= item.end_index:
                    if first:
                        labels[index] = f"B-{item.label}"
                        first = False
                    else:
                        labels[index] = f"I-{item.label}"
        return self.encoder.encode(labels)
    

class Model01TransformerModelTokenizationProcessor(BaseTokenizationProcessor[TransformerTokenizedDataset]):

    def __init__(self, tokenizer:TextTokenizer, intent_encoder:IntentClassificationEncoder, entity_encoder:BIOLabelEncoder):
        self.tokenizer = tokenizer
        self.intent_encoder = intent_encoder
        self.entity_encoder = entity_encoder
        self.entity_labeler = BIOLabeler(entity_encoder)

    def process(self, sample:ProcessedDataset) -> TransformerTokenizedDataset:
        tokens = self.tokenizer.tokenize(sample.text)

        return TransformerTokenizedDataset(
            text=sample.text,
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            offset_mapping=tokens["offset_mapping"],
            intent_labels=self.create_intent_labels(sample),
            ner_labels=self.create_entity_labels(sample, tokens["offset_mapping"]),
        )

    def create_intent_labels(self, sample:ProcessedDataset) -> list[int]:
        labels = [
            intent.label for intent in sample.intents
        ]
        return self.intent_encoder.encode(labels)

    def create_entity_labels(self, sample:ProcessedDataset, offsets:list[tuple[int, int]]) -> list[int]:
        entities:list[EntitySpan] = []
        for intent in sample.intents:
            entities.extend(intent.entities)

        return self.entity_labeler.create(entities, offsets)


class Model02TokenizationProcessor(BaseTokenizationProcessor[SpanIntentTokenizedDataset]):

    def __init__(self, tokenizer:TextTokenizer, intent_encoder:BIOLabelEncoder, entity_encoder:BIOLabelEncoder):
        self.tokenizer = tokenizer
        self.intent_labeler = BIOLabeler(intent_encoder)
        self.entity_labeler = BIOLabeler(entity_encoder)

    def process(self, sample):
        tokens = self.tokenizer.tokenize(sample.text)

        return SpanIntentTokenizedDataset(
            text=sample.text,
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            offset_mapping=tokens["offset_mapping"],
            intent_labels=self.intent_labeler.create(sample.intents, tokens["offset_mapping"]),
            ner_labels=self.create_entity_labels(sample, tokens["offset_mapping"]),
        )

    def create_entity_labels(self, sample:ProcessedDataset, offsets:list[tuple[int, int]]) -> list[int]:
        entities:list[EntitySpan] = []
        for intent in sample.intents:
            entities.extend(intent.entities)

        return self.entity_labeler.create(entities, offsets)


import json

from banking_nlu.dataprocessors.tokenizers import TextTokenizer


class Model03TokenizationProcessor:

    def __init__(
        self,
        tokenizer: TextTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process(self, item):

        target = {
            "text": item.text,
            "intents": [
                {
                    "label": intent.label,
                    "entities": [
                        {
                            "label": entity.label,
                            "value": entity.value,
                        }
                        for entity in intent.entities
                    ],
                }
                for intent in item.intents
            ],
        }

        target_json = json.dumps(
            target,
            ensure_ascii=False,
        )

        prompt = f"""Extract the intents and entities from this banking request.

Input:
{item.text}

Return JSON only.

Answer:
"""

        prompt_tokens = self.tokenizer.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        target_tokens = self.tokenizer.tokenizer(
            target_json,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = (
            prompt_tokens["input_ids"]
            + target_tokens["input_ids"]
        )

        attention_mask = (
            prompt_tokens["attention_mask"]
            + target_tokens["attention_mask"]
        )

        labels = (
            [-100] * len(prompt_tokens["input_ids"])
            + target_tokens["input_ids"]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": item.text,
        }