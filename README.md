# atcm-banking-nlu-pipeline
NLU model training pipeline for atcm-banking-nlu model

# Project Setup

Some project resources are excluded from Git because they are generated files, large model files, evaluation outputs, or environment-specific configurations.

After cloning the repository, create the required folders and files manually before running training, testing, or evaluation.

## Required Project Structure

```text
project-root/
│
├── _saved/
│   └── (trained model checkpoints)
│
├── _datasets/
│   └── testing_208.json
│       (testing dataset)
│   └── training_1014.json
│       (training dataset)
│
├── _evaluations/
│   └── (evaluation result files)
│
├── metadata/
│   ├── intents.json
│   └── entities.json
│
└── .env
```

## Folder Description

### `_saved/`

Stores trained model checkpoints generated during training.

Example:

```text
_saved/
└── banking_nlu_model_02_xxxxx.pt
```

Configured by:

```env
SAVED_MODEL_PATH=./_saved
SAVED_MODEL_NAME_PREFIX=banking_nlu_model_02
```

---


Configured by:

```env
TRAINING_DATASIZE=1014
TRAINING_FILE=./datasets/training_1014.json
```

---

### `_datasets/`

Contains testing datasets used for model evaluation.

Required file:

```text
_datasets/
└── testing_208.json
└── training_1014.json
```

Configured by:

```env
TESTING_DATASIZE=208
TESTING_FILE=./_datasets/testing_${TESTING_DATASIZE}.json
```

---

### `metadata/`

Contains metadata files required for label mapping.

Required files:

```text
metadata/
├── intents.json
└── entities.json
```

These files are used to map intent labels and entity labels between datasets and model outputs.

Configured by:

```env
INTENT_META_FILE=./metadata/intents.json
ENTITY_META_FILE=./metadata/entities.json
```

---

### `_evaluations/`

Stores evaluation results generated after evaluating the model.

Example:

```text
_evaluations/
└── evaluation_banking_nlu_model_02_t208.json
```

Configured by:

```env
EVALUATION_FILE=./_evaluations/evaluation_${SAVED_MODEL_NAME_PREFIX}_t${TESTING_DATASIZE}.json
```

---

# Environment Configuration

Create a `.env` file in the project root:

```env
SAVED_MODEL_PATH=./_saved

TEST_PROMPT=ဝင်းစုစုမျိုးဆီ ၄၀၀၀၀၀၀ လွှဲပေးပါဦး

INTENT_THRESHOLD=0.5

EPOCHS=30

TRAINING_DATASIZE=1014
TRAINING_FILE=./datasets/training_1014.json

SAVED_MODEL_NAME_PREFIX=banking_nlu_model_02

INTENT_META_FILE=./metadata/intents.json
ENTITY_META_FILE=./metadata/entities.json

TESTING_DATASIZE=208
TESTING_FILE=./_datasets/testing_${TESTING_DATASIZE}.json

EVALUATION_FILE=./_evaluations/evaluation_${SAVED_MODEL_NAME_PREFIX}_t${TESTING_DATASIZE}.json
```

---

# Notes

- Folders starting with `_` (`_saved`, `_datasets`, `_evaluations`) are intentionally ignored by Git.
- Trained model checkpoints should not be committed because of their large size.
- Evaluation outputs are generated files and should not be committed.
- Each developer should create their own `.env` file after cloning the repository.
- Ensure all required folders and datasets exist before running training or evaluation.