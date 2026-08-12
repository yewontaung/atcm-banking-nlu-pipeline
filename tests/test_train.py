from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.utils import env


processor = DataPreProcessor()

data = processor.process_file(env.TRAIN_JSON)

for item in data:
    print(item.model_dump_json(indent=2))