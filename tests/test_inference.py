import argparse
import torch

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from banking_nlu.models.model_01_transformer_model.model import BankingNLUTransformerModel

from banking_nlu.dataprocessors.encoders.classification import IntentClassificationEncoder
from banking_nlu.dataprocessors.encoders.bio import LabelBIOEncoder

from banking_nlu.dataprocessors.postprocessors.logit_mappers.classification import ClassificationLogitMapper
from banking_nlu.dataprocessors.postprocessors.logit_mappers.bio import BIOCombiner

from banking_nlu.utils import env
from banking_nlu.utils.loader import load_modelname
from banking_nlu.utils.schemas import TransformerModelOutput


class NLUInference:
    def __init__(self):
        self.DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )        
        self.intent_encoder = IntentClassificationEncoder.from_file(
            "./metadata/intents.json"
        )
        self.entity_encoder = LabelBIOEncoder.from_file(
            "./metadata/entities.json"
        )

        self.model = BankingNLUTransformerModel(
            model_name="xlm-roberta-base",
            intent_count=self.intent_encoder.no_of_lables,
            entity_count=self.entity_encoder.no_of_labels
        )
        checkpoint_path = (
            f"{env.SAVED_MODEL_PATH}/{load_modelname()}"
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )
        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.model.to(self.DEVICE)
        self.model.eval()

        self.tokenizer:PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(
                "xlm-roberta-base"
            )
        )

        self.mapper = ClassificationLogitMapper(
            intent_encoder=self.intent_encoder,
            entity_encoder=self.entity_encoder,
            intent_threshold=float(
                env.INTENT_THRESHOLD
            )
        )

        self.entity_combiner = BIOCombiner()


    def predict(self, text:str):

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True
        )


        with torch.no_grad():

            output:TransformerModelOutput = self.model(
                input_ids=encoded["input_ids"].to(self.DEVICE),
                attention_mask=encoded["attention_mask"].to(self.DEVICE)
            )

        intents = self.mapper.map_intents(
            output.intent_logits
        )
        entity_tokens = self.mapper.map_entities(
            output.entity_logits,
            text,
            encoded["offset_mapping"][0]
        )

        entities = self.entity_combiner.combine(
            text,
            entity_tokens
        )


        return {
            "text": text,
            "intents": intents,
            "entities": entities
        }



def parse_args():

    parser = argparse.ArgumentParser(
        description="Banking NLU inference"
    )


    parser.add_argument(
        "-m",
        "--message",
        type=str,
        required=False,
        default=env.TEST_PROMPT,
        help="Input Burmese sentence"
    )


    return parser.parse_args()



def main():

    args = parse_args()


    inference = NLUInference()


    result = inference.predict(
        args.message
    )


    print(
        f"TEXT : {result['text']}"
    )


    print(
        "==== intent prediction ===="
    )

    print(
        result["intents"]
    )


    print(
        "==== entity prediction ===="
    )

    print(
        result["entities"]
    )



if __name__ == "__main__":
    main()