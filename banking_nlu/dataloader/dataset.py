from torch.utils.data import Dataset

from banking_nlu.utils.schemas import TransformerTokenizedDataset, SpanIntentTokenizedDataset

class NLUDataset(Dataset):

    def __init__(self, samples:list[TransformerTokenizedDataset | SpanIntentTokenizedDataset]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index:int):
        result = self.samples[index]
        return {
            "input_ids": result.input_ids,
            "attention_mask": result.attention_mask,
            "intent_labels": result.intent_labels,
            "ner_labels": result.ner_labels
        }

class Model03Dataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        result = self.data[index]

        return {
            "input_ids": result.input_ids,
            "attention_mask": result.attention_mask,
            "labels": result.labels,
        }