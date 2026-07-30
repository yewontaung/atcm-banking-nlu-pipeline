import torch

from seqeval.metrics import accuracy_score, classification_report
from banking_nlu.dataprocessors.encoders.bio import LabelBIOEncoder
from banking_nlu.dataprocessors.preprocessors import DataPreProcessor
from banking_nlu.dataprocessors.tokenizers import Model02TokenizationProcessor, TextTokenizer
from banking_nlu.inference.mappers.model_02 import Model02PredictionMapper
from banking_nlu.models.model_02_token_intent_transformer_model.model import Model02BankingNLUTransformerModel
from banking_nlu.utils import env
from banking_nlu.utils.loader import load_modelname, load_saved_model
from banking_nlu.utils.schemas import Model02LogitOutput

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

intent_encoder = LabelBIOEncoder.from_file(env.INTENT_META_FILE)
entity_encoder = LabelBIOEncoder.from_file(env.ENTITY_META_FILE)

processor = DataPreProcessor()

processed = processor.process_file(env.TESTING_FILE)

def evaluate():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenization_processor = Model02TokenizationProcessor(
        tokenizer=TextTokenizer(tokenizer),
        intent_encoder=intent_encoder,
        entity_encoder=entity_encoder
    )

    tokenized = [tokenization_processor.process(item) for item in processed]
    true_intents = []
    true_entities = []

    pred_intents = []
    pred_entities = []

    mapper = Model02PredictionMapper(intent_encoder, entity_encoder)

    model = Model02BankingNLUTransformerModel(
        model_name="xlm-roberta-base",
        intent_count=intent_encoder.no_of_labels,
        entity_count=entity_encoder.no_of_labels
    )

    saved_model = load_saved_model(model, saved_model_path=f"{env.SAVED_MODEL_PATH}/{load_modelname()}", device=DEVICE)
    saved_model.eval()

    for item in tokenized:
        true_intents.append(intent_encoder.decode(item.intent_labels))
        true_entities.append(entity_encoder.decode(item.ner_labels))

        token = tokenizer(item.text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        input_ids = token["input_ids"].to(DEVICE)
        attention_mask = token["attention_mask"].to(DEVICE)
        offset_mapping = token["offset_mapping"][0]
        with torch.no_grad():
            outputs:Model02LogitOutput = saved_model(input_ids, attention_mask)
        pred_intents.append(mapper.intent_mapper.decode(outputs.intent_logits, text=item.text, offset_mapping=offset_mapping))
        pred_entities.append(mapper.entity_mapper.decode(outputs.entity_logits, text=item.text, offset_mapping=offset_mapping))

    print("==== TRUE ====")
    print("++++++ Intent ++++++")
    print(true_intents)
    print("++++++ Entity ++++++")
    print(true_entities)
    print("==== xxxx ====")
    print("==== Pred ====")
    print("++++++ Intent ++++++")
    print(pred_intents)
    print("++++++ Entity ++++++")
    print(pred_entities)
    print("==== xxxx ====")

    intent_report = classification_report(true_intents, pred_intents)
    print("===== Intent Report =====")
    print(intent_report)
    intent_accuracy = accuracy_score(true_intents, pred_intents)
    print("======= Intent Accuracy ========")
    print(intent_accuracy)

if __name__ == "__main__":
    evaluate()