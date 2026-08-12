from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.utils import env


processor = DataPreProcessor()

error = []

train = processor.load_file(env.TRAIN_JSON)
val = processor.load_file(env.VALIDATE_JSON)

for item in train:
    try:
        processor.process_item(item)
    except Exception as e:
        print(e)
        error.append(item)

print("Train error")
print(item)

print("Validation error")
for item in val:
    try:
        processor.process_item(item)
    except:
        error.append(item)
