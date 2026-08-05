from transformers import AutoTokenizer

from banking_nlu.dataprocessors.encoders.bio import BIOLabelEncoder
from banking_nlu.dataprocessors.tokenizers import Model02TokenizationProcessor, TextTokenizer
from banking_nlu.inference.mappers.model_02 import Model02PredictionMapper
from banking_nlu.models.model_02_token_intent_transformer_model.model import Model02BankingNLUTransformerModel
from banking_nlu.utils import env
from banking_nlu.utils.loader import load_saved_model
from banking_nlu.utils.schemas import EntitySpan, IntentSpan, Model02LogitOutput, ProcessedDataset

PROMPT = "09450001122 ထဲကို ၃၀၀၀ ဖုန်းကတ်ဖြည့်ပေးပါဦး။"

def output():
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    intent_encoder = BIOLabelEncoder.from_file(env.INTENT_META_FILE)
    entity_encoder = BIOLabelEncoder.from_file(env.ENTITY_META_FILE)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    mapper = Model02PredictionMapper(intent_encoder, entity_encoder)
    model = Model02BankingNLUTransformerModel(
        model_name="xlm-roberta-base",
        intent_count=intent_encoder.no_of_labels,
        entity_count=entity_encoder.no_of_labels
    )

    tokenized = tokenizer(PROMPT, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    saved_model = load_saved_model(model, f"{env.SAVED_MODEL_PATH}", DEVICE)
    saved_model.eval()

    with torch.no_grad():
        outputs:Model02LogitOutput = saved_model(tokenized["input_ids"].to(DEVICE), tokenized["attention_mask"].to(DEVICE))

    offset_mapping = tokenized["offset_mapping"][0]

    intents = mapper.intent_mapper.decode(logits=outputs.intent_logits, text=PROMPT, offset_mapping=offset_mapping)
    entities = mapper.entity_mapper.decode(logits=outputs.entity_logits, text=PROMPT, offset_mapping=offset_mapping)

    print("===== intents ======")
    print(intents)

    print("===== entities ======")
    print(entities)

def tokenized_output():
    tokenizer = TextTokenizer(AutoTokenizer.from_pretrained("xlm-roberta-base"))
    intent_encoder = BIOLabelEncoder.from_file(env.INTENT_META_FILE)
    entity_encoder = BIOLabelEncoder.from_file(env.ENTITY_META_FILE)
    tokenization_processor = Model02TokenizationProcessor(
        tokenizer=tokenizer,
        intent_encoder=intent_encoder,
        entity_encoder=entity_encoder
    )
    result = tokenization_processor.process(ProcessedDataset(
        text=PROMPT,
        intents=[IntentSpan(
            label="mobile_topup",
            start_index=0,
            end_index=44,
            entities=[
                EntitySpan(label="phone_number", start_index=0, end_index=11),
                EntitySpan(label="amount", start_index=18, end_index=22),
            ]
        )]
    ))

    print("==== Intents =====")
    print(intent_encoder.decode(result.intent_labels))
    print("==== Entities =====")
    print(entity_encoder.decode(result.ner_labels))

if __name__ == "__main__":
    output()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    tokenized_output()