def main():
    print("==== Model 01 Preview ====")
    import torch
    from transformers import AutoTokenizer

    from banking_nlu.dataprocessors.encoders.bio import BIOLabelEncoder
    from banking_nlu.dataprocessors.encoders.classification import IntentClassificationEncoder
    from banking_nlu.inference.mappers.model_01 import Model01PredictionMapper
    from banking_nlu.inference.models.model_01_inference import Model01Inference
    from banking_nlu.models.model_01_transformer_model.model import Model01BankingNLUTransformerModel
    from banking_nlu.utils import env
    from banking_nlu.utils.loader import load_saved_model

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    intent_encoder = IntentClassificationEncoder.from_file(env.INTENT_META_FILE)
    entity_encoder = BIOLabelEncoder.from_file(env.ENTITY_META_FILE)

    model = Model01BankingNLUTransformerModel(
        model_name="xlm-roberta-base",
        intent_count=intent_encoder.no_of_lables,
        entity_count=entity_encoder.no_of_labels,
    )

    saved_model = load_saved_model(model, env.SAVED_MODEL_PATH, DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    model_01_inference = Model01Inference(
        model=saved_model,
        intent_encoder=intent_encoder,
        entity_encoder=entity_encoder,
        DEVICE=DEVICE,
        mapper=Model01PredictionMapper(
            intent_encoder=intent_encoder,
            entity_encoder=entity_encoder
        ),
        tokenizer=tokenizer,
    )

    while True:
        text = input("Enter prompt: ")
        if text.lower() == "exit" or text.lower() == "0":
            break
        result = model_01_inference.predict(text)

        print(result)

    print("**************************")

if __name__ == "__main__":
    main()