from transformers import AutoTokenizer, SentencePieceBackend

from dataprocessors.encoders import EntityEncoder, IntentEncoder
from utils.schemas import ProcessedDataset, TokenizedDataset


class TokenizationProcessor:

    def __init__(self, model_name:str, intent_encoder:IntentEncoder, entity_encoder:EntityEncoder):
        self.tokenizer:SentencePieceBackend = AutoTokenizer.from_pretrained(model_name)
        self.intent_encoder = intent_encoder
        self.entity_encoder = entity_encoder

    def process(self, sample:ProcessedDataset) -> TokenizedDataset:
        tokens = self.tokenizer(
            sample.text,
            return_offsets_mapping=True,
            truncation=True,
            padding=False
        )

        return TokenizedDataset(
            text=sample.text,
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            intent_labels=self.create_intent_labels(sample),
            ner_labels=self.create_entity_labels(sample, tokens["offset_mapping"]),
        )

    def create_intent_labels(self, sample:ProcessedDataset) -> list[int]:
        labels = [
            intent.label for intent in sample.intents
        ]

        return self.intent_encoder.encode(labels)

    def create_entity_labels(self, sample:ProcessedDataset, offsets:list[tuple[int, int]]) -> list[int]:
        labels = ["O" for _ in offsets]
        for intent in sample.intents:
            for entity in intent.entities:
                first = True
                for index, (token_start, token_end) in enumerate(offsets):
                    if token_start == token_end:
                        continue
                    if token_start >= entity.start_index and token_end <= entity.end_index:
                        if first:
                            labels[index] = f"B-{entity.label}"
                            first = False
                        else:
                            labels[index] = f"I-{entity.label}"
        return self.entity_encoder.encode(labels)
