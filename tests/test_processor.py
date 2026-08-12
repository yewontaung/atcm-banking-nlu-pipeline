from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.utils import env


processor = DataPreProcessor()


train = processor.load_file(env.TRAIN_JSON)
val = processor.load_file(env.VALIDATE_JSON)

train_error = []

for item in train:
    try:
        processor.process_item(item)
    except Exception as e:
        print(e)
        train_error.append(item)

print("Train error")
print(train_error)

val_error = []

for item in val:
    try:
        processor.process_item(item)
    except:
        val_error.append(item)
print("Validation error")
print(val_error)
