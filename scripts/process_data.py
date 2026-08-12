import json

from sklearn.model_selection import train_test_split

from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.utils import env


print("======== Starting Script =========")

preprocessor = DataPreProcessor()

data_items = preprocessor.load_file(env.TRAINING_FILE)

items = [item.model_dump() for item in data_items]

RANDOM_STATE = 42

train_data, validation_data = train_test_split(
    items,
    test_size=float(env.VALIDATION_SIZE),
    random_state=RANDOM_STATE,
    shuffle=True,
)

with open(env.TRAIN_JSON, "w", encoding="utf-8") as file:
    json.dump(
        train_data,
        file,
        ensure_ascii=False,
        indent=2,
    )

with open(env.VALIDATE_JSON, "w", encoding="utf-8") as file:
    json.dump(
        validation_data,
        file,
        ensure_ascii=False,
        indent=2,
    )

print(f"Total:      {len(data_items)}")
print(f"Training:   {len(train_data)}")
print(f"Validation: {len(validation_data)}")