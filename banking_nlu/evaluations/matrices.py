from seqeval.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from banking_nlu.utils.schemas import EvaluationData, EvaluationResult, Model02EvaluationResult


def calculate_model02_matrices(evaluation_data:EvaluationData) -> Model02EvaluationResult:

    intent_truth = evaluation_data.intent_truth
    intent_evaluation = calculate_evaluation_result(intent_truth.grounded_truth, intent_truth.predicted_truth)

    entity_truth = evaluation_data.entity_truth
    entity_evaluation = calculate_evaluation_result(entity_truth.grounded_truth, entity_truth.predicted_truth)

    return Model02EvaluationResult(
        intent_evaluation=intent_evaluation,
        entity_evaluation=entity_evaluation
    )

def calculate_evaluation_result(grounded_truth, pred_truth) -> EvaluationResult:
    return EvaluationResult(
        report=classification_report(grounded_truth, pred_truth, output_dict=True),
        accuracy=accuracy_score(grounded_truth, pred_truth),
        precision=precision_score(grounded_truth, pred_truth),
        recall=recall_score(grounded_truth, pred_truth),
        f1=f1_score(grounded_truth, pred_truth)
    )
