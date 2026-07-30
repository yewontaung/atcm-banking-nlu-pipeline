import torch

from seqeval.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from banking_nlu.utils.schemas import EvaluationData, EvaluationResult, Model02EvaluationResult, Model02LogitOutput, SpanIntentTokenizedDataset, TruthResult


class Model02Evaluator:

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

    def get_evaluation_data(self, tokenized:list[SpanIntentTokenizedDataset]) -> EvaluationData:
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
                outputs:Model02LogitOutput = self.model(input_ids, attention_mask)
                pred_intents.append(self.mapper.intent_mapper.decode(outputs.intent_logits, text=item.text, offset_mapping=offset_mapping))
                pred_entities.append(self.mapper.entity_mapper.decode(outputs.entity_logits, text=item.text, offset_mapping=offset_mapping))

        return EvaluationData(
            intent_truth=TruthResult(grounded_truth=true_intents, predicted_truth=pred_intents),
            entity_truth=TruthResult(grounded_truth=true_entities, predicted_truth=pred_entities)
        )

    def calculate_matrices(self, evaluation_data:EvaluationData) -> Model02EvaluationResult:

        intent_truth = evaluation_data.intent_truth
        intent_evaluation = self.calculate_evaluation_result(intent_truth.grounded_truth, intent_truth.predicted_truth)

        entity_truth = evaluation_data.entity_truth
        entity_evaluation = self.calculate_evaluation_result(entity_truth.grounded_truth, entity_truth.predicted_truth)

        return Model02EvaluationResult(
            intent_evaluation=intent_evaluation,
            entity_evaluation=entity_evaluation
        )

    def calculate_evaluation_result(self, grounded_truth, pred_truth) -> EvaluationResult:
        return EvaluationResult(
            report=classification_report(grounded_truth, pred_truth, output_dict=True),
            accuracy=accuracy_score(grounded_truth, pred_truth),
            precision=precision_score(grounded_truth, pred_truth),
            recall=recall_score(grounded_truth, pred_truth),
            f1=f1_score(grounded_truth, pred_truth)
        )    