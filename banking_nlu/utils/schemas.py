from dataclasses import dataclass

from pydantic import BaseModel
from torch import Tensor


# ==========================================================
# Metadata schemas
# ==========================================================
class IntentMeta(BaseModel):
    intent_id:int
    label:str

class EntityMeta(BaseModel):
    entity_id:int
    label:str

# ==========================================================
# Raw exported dataset schemas
# ==========================================================
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

# ==========================================================
# Generic span schemas
# ==========================================================
class Span(BaseModel):
    label: str
    start_index: int
    end_index: int

class EntitySpan(Span):
    pass

class IntentSpan(Span):
    entities: list[EntitySpan]

class ProcessedDataset(BaseModel):
    text:str
    intents:list[IntentSpan]

# ==========================================================
# Tokenized common schema
# ==========================================================
class TokenizedDataset(BaseModel):
    """
    Common tokenized data shared by all models
    """
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    offset_mapping: list[tuple[int, int]] | None = None

# ==========================================================
# Model specific training samples
# ==========================================================
class TransformerTokenizedDataset(TokenizedDataset):
    """
    Model 1:
    Sentence-level intent classification
    + BIO entity extraction
    """
    intent_labels:list[int]
    ner_labels:list[int]

class SpanIntentTokenizedDataset(TokenizedDataset):
    """
    Model 2:
    Intent span extraction
    + BIO entity extraction
    """
    intent_labels: list[int]
    ner_labels: list[int]

# ==========================================================
# Model IO
# ==========================================================

@dataclass
class ModelInput:
    input_ids: Tensor
    attention_mask: Tensor

@dataclass
class TransformerModelInput(ModelInput):
    intent_labels: Tensor
    ner_labels: Tensor

@dataclass
class TokenIntentModelInput(ModelInput):
    intent_span_labels: Tensor
    ner_labels: Tensor

@dataclass
class ModelOutput:
    model_config = {
        "arbitrary_types_allowed": True
    }

@dataclass
class TransformerModelOutput(ModelOutput):
    intent_logits: Tensor
    entity_logits: Tensor

@dataclass
class Model02LogitOutput(ModelOutput):
    intent_logits: Tensor
    entity_logits: Tensor


# ==========================================================
# Prediction schemas
# ==========================================================
class ClassifiedIntentPrediction(BaseModel):
    prediction_id:int
    label:str
    confidence:float

class TokenPrediction(BaseModel):
    prediction_id:int
    label:str
    value:str
    confidence:float
    start_index:int
    end_index:int

class SpanPrediction(BaseModel):
    prediction_id:int
    label:str
    value:str
    confidence:float
    start_index:int
    end_index:int

# Final API response schemas
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