from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.utils import env


processor = DataPreProcessor()

train = processor.process_file(env.TRAIN_JSON)
val = processor.process_file(env.VALIDATION_SIZE)

print(train)
print(val)