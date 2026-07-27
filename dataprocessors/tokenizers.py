from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

from transformers import AutoTokenizer, SentencePieceBackend

from dataprocessors.encoders.bio import LabelBIOEncoder
from dataprocessors.encoders.classification import IntentClassificationEncoder
from utils.schemas import EntitySpan, IntentSpan, ProcessedDataset, Span, TokenizedDataset, TransformerTokenizedDataset, SpanIntentTokenizedDataset

R = TypeVar("R", bound=TokenizedDataset)

class TextTokenizer:

    def __init__(self, model_name:str):
        self.tokenizer:SentencePieceBackend = AutoTokenizer.from_pretrained(model_name)

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

    def __init__(self, encoder:LabelBIOEncoder):
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
    

class TransformerModelTokenizationProcessor(BaseTokenizationProcessor[TransformerTokenizedDataset]):

    def __init__(self, tokenizer:TextTokenizer, intent_encoder:IntentClassificationEncoder, entity_encoder:LabelBIOEncoder):
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


class SpanBasedModelTokenizationProcessor(BaseTokenizationProcessor[SpanIntentTokenizedDataset]):

    def __init__(self, tokenizer:TextTokenizer, intent_encoder:LabelBIOEncoder, entity_encoder:LabelBIOEncoder):
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
            intent_span_labels=self.intent_labeler.create(sample.intents, tokens["offset_mapping"]),
            ner_labels=self.create_entity_labels(sample, tokens["offset_mapping"]),
        )

    def create_entity_labels(self, sample:ProcessedDataset, offsets:list[tuple[int, int]]) -> list[int]:
        entities:list[EntitySpan] = []
        for intent in sample.intents:
            entities.extend(intent.entities)

        return self.entity_labeler.create(entities, offsets)
