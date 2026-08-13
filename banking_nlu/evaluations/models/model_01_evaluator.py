import torch

import sklearn.metrics as sk
from seqeval.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from banking_nlu.evaluations.matrices import make_json_serializable
from banking_nlu.utils.schemas import EvaluationData, EvaluationResult, Model01LogitOutput, Model02EvaluationResult, TransformerTokenizedDataset, TruthResult


class Model01Evaluator:

    def __init__(
            self, 
            model, 
            tokenizer,
            intent_encoder,
            entity_encoder,
            mapper,
            device):
        self.model = model
        self.tokenizer = tokenizer
        self.intent_encoder = intent_encoder
        self.entity_encoder = entity_encoder
        self.mapper = mapper
        self.device = device

    def get_evaluation_data(self, tokenized:list[TransformerTokenizedDataset]) -> EvaluationData:
        true_intents = []
        true_entities = []

        pred_intents = []
        pred_entities = []

        self.model.eval()

        for item in tokenized:
            true_intents.append(self.intent_encoder.decode(item.intent_labels))
            true_entities.append(self.entity_encoder.decode(item.ner_labels))
    
            token = self.tokenizer(item.text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
            input_ids = token["input_ids"].to(self.device)
            attention_mask = token["attention_mask"].to(self.device)
            offset_mapping = token["offset_mapping"][0]
            with torch.no_grad():
                outputs:Model01LogitOutput = self.model(input_ids, attention_mask)
                pred_intents.append(self.mapper.decode_intents(outputs.intent_logits))
                pred_entities.append(self.mapper.decode_entities(outputs.entity_logits, offset_mapping=offset_mapping))

        return EvaluationData(
            intent_truth=TruthResult(grounded_truth=true_intents, predicted_truth=pred_intents),
            entity_truth=TruthResult(grounded_truth=true_entities, predicted_truth=pred_entities)
        )

    def calculate_matrices(self, evaluation_data):

        intent_truth = evaluation_data.intent_truth

        intent_evaluation = self.calculate_intent_evaluation(
            intent_truth.grounded_truth,
            intent_truth.predicted_truth
        )

        entity_truth = evaluation_data.entity_truth

        entity_evaluation = self.calculate_entity_evaluation(
            entity_truth.grounded_truth,
            entity_truth.predicted_truth
        )

        return Model02EvaluationResult(
            intent_evaluation=intent_evaluation,
            entity_evaluation=entity_evaluation
        )

    def calculate_intent_evaluation(self, grounded_truth, pred_truth) -> EvaluationResult:
        return EvaluationResult(
            report=make_json_serializable(sk.classification_report(grounded_truth, pred_truth, output_dict=True)),
            accuracy=sk.accuracy_score(grounded_truth, pred_truth),
            precision=sk.precision_score(grounded_truth, pred_truth),
            recall=sk.recall_score(grounded_truth, pred_truth),
            f1=sk.f1_score(grounded_truth, pred_truth)
        )

    def calculate_entity_evaluation_result(self, grounded_truth, pred_truth) -> EvaluationResult:
        return EvaluationResult(
            report=make_json_serializable(classification_report(grounded_truth, pred_truth, output_dict=True)),
            accuracy=accuracy_score(grounded_truth, pred_truth),
            precision=precision_score(grounded_truth, pred_truth),
            recall=recall_score(grounded_truth, pred_truth),
            f1=f1_score(grounded_truth, pred_truth)
        )