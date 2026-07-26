import torch
from torch.nn.utils.rnn import pad_sequence


class NLUCollator:

    def __init__(self, pad_token_id:int, ner_pad_id:int = -100):
        self.pad_token_id = pad_token_id
        self.ner_pad_id = ner_pad_id

    def __call__(self, batch:list[dict]):
        input_ids = [
            torch.tensor(item["input_ids"], dtype=torch.long) for item in batch
        ]

        attention_mask = [
            torch.tensor(item["attention_mask"]) for item in batch
        ]

        ner_labes = [
            torch.tensor(item["ner_labels"], dtype=torch.long) for item in batch
        ]

        intent_labes = [
            torch.tensor(item["intent_labels"], dtype=torch.long) for item in batch
        ]

        return {
            "input_ids": pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.pad_token_id
            ),
            "attention_mask": pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0
            ),
            "intent_labels": torch.stack(intent_labes),
            "ner_labels": pad_sequence(
                ner_labes,
                batch_first=True,
                padding_value=self.ner_pad_id
            )
        }


