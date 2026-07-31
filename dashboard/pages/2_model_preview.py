import streamlit as st

from dashboard.services.loader import load_predictor

st.title("Model Preview")

st.header("Test the model.")

predictor = load_predictor()

prompt = st.text_area(label="Prompt", placeholder="Ask our model to do something with banking.", height=20)

col1, col2 = st.columns([1, 4])

with col1:
    predict = st.button("Predict", use_container_width=True)

if predict:
    if not prompt.strip():
        st.warning("Please enter a message.")
    else:
        st.divider()
        st.subheader("Prediction Result")
        prediction = predictor.predict(prompt.strip())

        st.json(prediction.model_dump_json(indent=2))
        