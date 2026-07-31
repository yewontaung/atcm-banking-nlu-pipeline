import streamlit as st
import pandas as pd
import json

from banking_nlu.utils import env
from banking_nlu.utils.schemas import EvaluationResult, Model02EvaluationResult

@st.cache_resource
def load_evaluation(file_path:str):
    with open(file_path, encoding="utf-8") as file:
        result = json.load(file)
        return Model02EvaluationResult(**result)

def show_metrices(title, value):
    st.metric(label=title, value=f"{value:.4f}")

model02_evaluation = load_evaluation(env.EVALUATION_FILE)

st.title("Banking NLU Model Evaluation")

def show_evaluations(header, evaluation:EvaluationResult):
    st.header(header)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metrices("Accuracy", evaluation.accuracy)
    with col2:
        show_metrices("Precision", evaluation.precision)
    with col3:
        show_metrices("Recall", evaluation.recall)
    with col4:
        show_metrices("F1 Score", evaluation.f1)

    intent_report = pd.DataFrame(evaluation.report).transpose()

    st.subheader(f"{header} Report")
    st.dataframe(
        intent_report,
        use_container_width=True
    )

st.header("Dataset Information")
col1, col2 = st.columns(2)
with col1:
    st.metric("Training Dataset", env.TRAINING_DATASIZE)
with col2:
    st.metric("Testing Dataset", env.TESTING_DATASIZE)


show_evaluations("Intent Classification Evaluation", model02_evaluation.intent_evaluation)
show_evaluations("Entity Classification Evaluation", model02_evaluation.entity_evaluation)
