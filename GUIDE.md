# Banking NLU Model Library Guide

## Overview

`banking-nlu` is an inference library for the Banking NLU model.

The project internally supports training multiple model versions, but this library exposes the released model for prediction.

The model supports:

- Intent classification
- Named Entity Recognition (NER)
- Structured prediction output

The current model is based on:

- Transformer: `XLM-RoBERTa`
- Tasks:
  - Multi-label intent classification
  - Token-level entity extraction

---

# Installation

Install the library using pip:

```bash
pip install banking-nlu
```

---

# Required Files

The predictor requires:

1. Trained model files
2. Intent metadata
3. Entity metadata

The directory structure should look like:

```
model/
├── model.pt
├── tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── sentencepiece.bpe.model
│
metadata/
├── intents.json
└── entities.json
```

---

# Download Metadata

Download the required metadata files:

- `intents.json`
- `entities.json`

from:

https://github.com/yewontaung/atcm-banking-nlu-pipeline/tree/main/metadata

These files are required to map model output indexes into readable intent and entity labels.

---

# Download Trained Model

Download the trained model artifact from:

```
<Model download location>
```

After downloading, extract the model files.

Example:

```
saved_models/
└── banking_nlu_model_v1/
    ├── model.pt
    └── tokenizer/
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── sentencepiece.bpe.model
```

---

# Creating Predictor

```python
from banking_nlu import BankingNLUPredictor


predictor = BankingNLUPredictor.model_predictor(
    model_name="xlm-roberta-base",
    saved_model_path="./saved_models/banking_nlu_model_v1",
    intent_metadata_path="./metadata/intents.json",
    entity_metadata_path="./metadata/entities.json",
    device="cpu",
)
```

### Parameters

| Parameter | Description |
|---|---|
| `model_name` | Base transformer model name |
| `saved_model_path` | Path to trained model directory |
| `intent_metadata_path` | Path to intent mapping metadata |
| `entity_metadata_path` | Path to entity mapping metadata |
| `device` | Inference device (`cpu` or `cuda`) |

---

# Using Predictor

```python
text = "အောင်အောင်ကို ၇၀၀၀၀ လွှဲလိုက်ပါ"

result = predictor.predict(text)

print(result)
```

---

# Prediction Result

The prediction result will have the following structure:

```json
{
    "text": "အောင်အောင်ကို ၇၀၀၀၀ လွှဲလိုက်ပါ",
    "intents": [
        {
            "label": "transfer_money",
            "confidence": 0.966,
            "entities": [
                {
                    "label": "receiver",
                    "value": "အောင်အောင်"
                },
                {
                    "label": "amount",
                    "value": "၇၀၀၀၀"
                }
            ]
        }
    ]
}
```

---

# Inference Pipeline

The library internally performs:

```
Input Text

    |
    v

Tokenizer

    |
    v

XLM-RoBERTa Model

    |
    v

Intent & Entity Logits

    |
    v

Logits Mapping

    |
    v

Entity Binding

    |
    v

Structured Prediction Result
```

---

# Notes

- The Python package contains only inference code.
- Model weights are distributed separately because of their large size.
- Metadata files must match the trained model version.
- The same tokenizer used during training must be used during inference.
- Different model versions may provide different prediction results.