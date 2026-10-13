"""Streamlit demo for BANKING77 customer-query intent routing."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import soundfile as sf
import streamlit as st
import torch
from transformers import pipeline

from banking77_core import (
    DEFAULT_MODEL_DIR,
    HuggingFaceEmbeddingRetriever,
    HuggingFaceIntentClassifier,
    load_banking77,
    make_reference_examples,
    model_weights_exist,
)

HUB_MODEL_ID = os.environ.get("BANKING77_MODEL_ID", "").strip()
MODEL_REF = HUB_MODEL_ID or str(DEFAULT_MODEL_DIR)
LOCAL_MODEL_CONFIGURED = bool(HUB_MODEL_ID) or model_weights_exist(DEFAULT_MODEL_DIR)


@st.cache_resource(show_spinner="Loading the fine-tuned intent model...")
def get_classifier(model_ref: str) -> HuggingFaceIntentClassifier:
    return HuggingFaceIntentClassifier(model_ref)


@st.cache_resource(show_spinner="Loading BANKING77 and building the retrieval index...")
def get_retriever() -> tuple[HuggingFaceEmbeddingRetriever, list[str]]:
    dataset, class_names = load_banking77()
    texts, labels = make_reference_examples(dataset["train"])
    retriever = HuggingFaceEmbeddingRetriever(texts, labels, class_names)
    return retriever, class_names


@st.cache_resource(show_spinner="Loading the speech-to-text model...")
def get_transcriber() -> Any:
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device=0 if torch.cuda.is_available() else -1,
    )


def display_predictions(
    predictions: list[dict[str, Any]],
    score_label: str,
) -> None:
    if not predictions:
        st.error("The selected pipeline returned no predictions.")
        return
    st.subheader(str(predictions[0]["intent"]).replace("_", " ").title())
    st.caption(f"{score_label}: {float(predictions[0]['score']):.4f}")
    if len(predictions) > 1:
        st.write("Other likely intents")
        for prediction in predictions[1:]:
            st.write(
                f"- {str(prediction['intent']).replace('_', ' ').title()} "
                f"({score_label.lower()}: {float(prediction['score']):.4f})"
            )


def render_query_interface(selected_mode: str) -> None:
    voice_mode = selected_mode == "Speak my message"

    if voice_mode:
        st.write(
            "Record your English enquiry. Whisper Tiny will transcribe it, then "
            "the fine-tuned classifier will predict its intent."
        )
        st.caption(
            "Tip: Click Record, speak your enquiry, click Stop, then click "
            "Classify message."
        )
        recorded_audio = st.audio_input(
            "Record a banking enquiry",
            sample_rate=16000,
        )
    else:
        query = st.text_area(
            "Customer message",
            placeholder="For example: I don't recognize a card payment on my account.",
            max_chars=2000,
        )
    if not st.button("Classify message", type="primary"):
        return
    if selected_mode != "Find similar examples" and not LOCAL_MODEL_CONFIGURED:
        st.error(
            "This mode requires the fine-tuned classifier. Configure "
            "`BANKING77_MODEL_ID` or run `python train.py` first."
        )
        return

    if voice_mode:
        if recorded_audio is None:
            st.warning("Record an enquiry before classifying.")
            return
        try:
            audio, sample_rate = sf.read(
                BytesIO(recorded_audio.getvalue()),
                dtype="float32",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"Could not read the recorded audio: {exc}")
            return
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        with st.spinner("Transcribing your recording..."):
            result = get_transcriber()(
                {"array": audio, "sampling_rate": sample_rate},
                generate_kwargs={"language": "english", "task": "transcribe"},
            )
        message = str(result["text"]).strip()
        if not message:
            st.warning("No speech was transcribed. Please try recording again.")
            return
        st.text_area("Speech-to-text transcript", value=message, disabled=True)
    else:
        message = query.strip()
        if not message:
            st.warning("Enter a customer message before classifying.")
            return

    if selected_mode in ("Classify my message", "Speak my message"):
        classifier = get_classifier(MODEL_REF)
        predictions = classifier.predict_many([message], top_k=3)[0]
        display_predictions(predictions, "Model probability")
    else:
        retriever, _ = get_retriever()
        predictions = retriever.predict_many([message], top_k=3)[0]
        display_predictions(predictions, "Mean neighbor cosine similarity")


def main() -> None:
    st.set_page_config(
        page_title="Banking Query Intent Router",
        page_icon="🏦",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --hsbc-red: #DB0011;
            --hsbc-red-dark: #B0000E;
            --hsbc-ink: #1B1B1B;
            --hsbc-canvas: #F5F5F5;
        }
        [data-testid="stAppViewContainer"] {
            background-color: var(--hsbc-canvas);
        }
        [data-testid="stHeader"] {
            background-color: rgba(255, 255, 255, 0.92);
        }
        .block-container {
            max-width: 1120px;
            padding-top: 4rem;
            padding-bottom: 2.5rem;
        }
        h1 {
            color: var(--hsbc-ink);
            font-weight: 700;
            font-size: 2rem;
            line-height: 1.2;
        }
        .hsbc-brand-section {
            color: #555555;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            margin-bottom: 0.25rem;
        }
        input[type="radio"] {
            accent-color: var(--hsbc-red);
        }
        div.stButton > button[kind="primary"] {
            background-color: var(--hsbc-red);
            border-color: var(--hsbc-red);
            color: #FFFFFF;
            font-weight: 600;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: var(--hsbc-red-dark);
            border-color: var(--hsbc-red-dark);
            color: #FFFFFF;
        }
        div[data-testid="stTextArea"] textarea:focus {
            border-color: var(--hsbc-red);
            box-shadow: 0 0 0 1px var(--hsbc-red);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logo_column, title_column = st.columns([1, 4])
    with logo_column:
        st.image(
            str(Path(__file__).resolve().parent / "assets" / "hsbc-logo-2018.svg"),
            width=180,
        )
    st.title("HSBC Banking Query Intent Router")

    if not LOCAL_MODEL_CONFIGURED:
        st.info(
            "The fine-tuned classifier is not available. Set `BANKING77_MODEL_ID` "
            "to a Hugging Face model repository containing the fine-tuned model, "
            "or run `python train.py` to create it locally. You can still try the "
            "pretrained retrieval baseline."
        )

    selected_mode = st.radio(
        "How would you like to submit your query?",
        [
            "Classify my message",
            "Find similar examples",
            "Speak my message",
        ],
        index=0 if LOCAL_MODEL_CONFIGURED else 1,
    )
    mode_explanations = {
        "Classify my message": (
            "Fine-tuned classifier: DistilBERT predicts one of BANKING77's 77 "
            "intent categories."
        ),
        "Find similar examples": (
            "Pretrained retrieval: finds the closest BANKING77 examples and "
            "uses their intent."
        ),
        "Speak my message": (
            "Voice-to-text: Whisper Tiny transcribes your recording, then the "
            "fine-tuned classifier predicts its intent."
        ),
    }
    st.caption(mode_explanations[selected_mode])
    render_query_interface(selected_mode)

    st.markdown(
        "<div style='text-align: center; color: #666666; font-size: 1rem; "
        "margin-top: 4rem;'>© 2026 Group02 | DistilBERT classifies BANKING77 "
        "intents, all-MiniLM-L6-v2 retrieves similar examples, and Whisper Tiny "
        "transcribes spoken enquiries to support online-banking query routing."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
