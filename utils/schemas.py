from dataclasses import dataclass

from pydantic import BaseModel
from torch import Tensor


# Metadata format
class IntentMeta(BaseModel):
    intent_id:int
    label:str

class EntityMeta(BaseModel):
    entity_id:int
    label:str

# Exported dataset structure

class ExportedEntity(BaseModel):
    datasetintentner_id:int
    ner_id:int
    label:str
    start_index:int
    end_index:int

class ExportedIntent(BaseModel):
    datasetintent_id:int
    intent_id:int
    label:str
    start_index:int
    end_index:int
    entities:list[ExportedEntity]

class ExportedDataset(BaseModel):
    dataset_id:int
    text:str
    intents:list[ExportedIntent]

# Processed dataset structure

class EntitySpan(BaseModel):
    label:str
    start_index:int
    end_index:int

class IntentSpan(BaseModel):
    label:str
    start_index:int
    end_index:int
    entities:list[EntitySpan]

class ProcessedDataset(BaseModel):
    text:str
    intents:list[IntentSpan]

# Tokenized dataset structure
class TokenizedDataset(BaseModel):
    text:str
    input_ids:list[int]
    attention_mask:list[int]
    intent_labels:list[int]
    ner_labels:list[int]

# Model schemas
@dataclass
class ModelInput:
    input_ids: Tensor
    attention_mask: Tensor
    intent_labels: Tensor
    ner_labels: Tensor

@dataclass
class ModelOutput:
    intent_logits: Tensor
    entity_logits: Tensor

    model_config = {
        "arbitrary_types_allowed": True
    }


# Prediction schemas

class IntentPrediction(BaseModel):
    prediction_id:int
    label:str
    confidence:float

class EntityPrediction(BaseModel):
    prediction_id:int
    label:str
    value:str
    confidence:float
    start_index:int
    end_index:int

class PredictedEntity(BaseModel):
    label:str
    value:str

class PredictedIntent(BaseModel):
    label:str
    confidence:float
    entities:list[PredictedEntity]

class ModelPrediction(BaseModel):
    text:str
    intents:list[PredictedIntent]