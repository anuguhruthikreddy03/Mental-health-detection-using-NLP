import streamlit as st
import numpy as np
import joblib
import torch
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from gensim.models import KeyedVectors

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Mental Health Detection System",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BERT_PATH = os.path.join(BASE_DIR, "bert_model")
XGB_PATH = os.path.join(BASE_DIR, "mental_health_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
FASTTEXT_PATH = os.path.join(BASE_DIR, "fasttext_model.kv")

VECTOR_SIZE = 300

# ==========================================
# LOAD BERT
# ==========================================

@st.cache_resource
def load_bert():

    if not os.path.exists(BERT_PATH):
        return None, None, "bert_model folder not found"

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            BERT_PATH
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            BERT_PATH
        )

        return tokenizer, model, None

    except Exception as e:

        return None, None, str(e)

# ==========================================
# LOAD XGBOOST
# ==========================================

@st.cache_resource
def load_xgb():

    missing_files = []

    if not os.path.exists(XGB_PATH):
        missing_files.append("mental_health_model.pkl")

    if not os.path.exists(ENCODER_PATH):
        missing_files.append("label_encoder.pkl")

    if not os.path.exists(FASTTEXT_PATH):
        missing_files.append("fasttext_model.kv")

    if len(missing_files) > 0:

        return None, None, None, missing_files

    try:

        model = joblib.load(
            XGB_PATH
        )

        encoder = joblib.load(
            ENCODER_PATH
        )

        fasttext_model = KeyedVectors.load(
            FASTTEXT_PATH
        )

        return model, encoder, fasttext_model, []

    except Exception as e:

        return None, None, None, [str(e)]

# ==========================================
# TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):

    text = text.lower()

    words = text.split()

    return words

# ==========================================
# FASTTEXT VECTORIZATION
# ==========================================

def vectorize_text(tokens, model):

    vectors = [
        model[word]
        for word in tokens
        if word in model.key_to_index
    ]

    if len(vectors) > 0:

        return np.mean(
            vectors,
            axis=0
        )

    return np.zeros(VECTOR_SIZE)

# ==========================================
# LOAD MODELS
# ==========================================

bert_tokenizer, bert_model, bert_error = load_bert()

xgb_model, encoder, fasttext_model, xgb_errors = load_xgb()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    [
        "BERT",
        "XGBoost"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Debug")

st.sidebar.write(
    "bert_model:",
    os.path.exists(BERT_PATH)
)

st.sidebar.write(
    "mental_health_model.pkl:",
    os.path.exists(XGB_PATH)
)

st.sidebar.write(
    "label_encoder.pkl:",
    os.path.exists(ENCODER_PATH)
)

st.sidebar.write(
    "fasttext_model.kv:",
    os.path.exists(FASTTEXT_PATH)
)

# ==========================================
# TITLE
# ==========================================

st.title("🧠 Mental Health Detection System")

st.markdown(
"""
This application predicts depression-related indicators using:

- 🤖 BERT
- ⚡ XGBoost + FastText
"""
)

# ==========================================
# INPUT
# ==========================================

user_text = st.text_area(
    "Enter Text",
    height=220,
    placeholder="Type your thoughts here..."
)

# ==========================================
# PREDICT
# ==========================================

if st.button("Predict"):

    if not user_text.strip():

        st.warning(
            "Please enter some text."
        )

    else:

        # ==================================
        # BERT
        # ==================================

        if model_choice == "BERT":

            if bert_model is None:

                st.error(
                    f"BERT Loading Error: {bert_error}"
                )

                st.stop()

            inputs = bert_tokenizer(
                user_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():

                outputs = bert_model(
                    **inputs
                )

                probs = torch.softmax(
                    outputs.logits,
                    dim=1
                )

                pred = torch.argmax(
                    probs,
                    dim=1
                ).item()

                confidence = (
                    probs.max().item() * 100
                )

            label_map = {
                0: "Not Depressed",
                1: "Depressed"
            }

            label = label_map.get(
                pred,
                str(pred)
            )

        # ==================================
        # XGBOOST
        # ==================================

        else:

            if xgb_model is None:

                st.error(
                    "❌ XGBoost files missing."
                )

                st.write("Missing/Error Files:")

                for err in xgb_errors:

                    st.write(f"• {err}")

                st.stop()

            tokens = preprocess_text(
                user_text
            )

            vector = vectorize_text(
                tokens,
                fasttext_model
            )

            pred = xgb_model.predict(
                [vector]
            )[0]

            confidence = (
                np.max(
                    xgb_model.predict_proba(
                        [vector]
                    )
                ) * 100
            )

            try:

                label = encoder.inverse_transform(
                    [pred]
                )[0]

            except:

                label_map = {
                    0: "Not Depressed",
                    1: "Depressed"
                }

                label = label_map.get(
                    pred,
                    str(pred)
                )

        # ==================================
        # RESULT
        # ==================================

        st.markdown("---")

        st.subheader("Prediction Result")

        if str(label).lower() in [
            "depressed",
            "1"
        ]:

            st.error(
                "⚠️ Depression Indicators Detected"
            )

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

            st.markdown(
                """
                ### 💡 Recommendations

                • Talk to someone you trust

                • Maintain healthy sleep habits

                • Exercise regularly

                • Stay socially connected

                • Practice mindfulness

                • Seek professional help if needed
                """
            )

        else:

            st.success(
                "✅ No Significant Depression Indicators Detected"
            )

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

            st.markdown(
                """
                ### 🌟 Positive Mental Wellness Tips

                • Continue healthy routines

                • Stay connected with friends and family

                • Practice gratitude

                • Exercise regularly

                • Maintain work-life balance

                • Continue self-care activities
                """
            )

        st.markdown("---")

        st.write("### Selected Model")
        st.write(model_choice)

        st.write("### Prediction")
        st.write(label)

        st.write("### Confidence")
        st.write(f"{confidence:.2f}%")