"""Streamlit demo for BANKING77 customer-query intent routing."""

from __future__ import annotations

import os
from io import BytesIO
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


def main() -> None:
    st.set_page_config(page_title="Banking Query Intent Router", page_icon="🏦")
    st.title("Banking Query Intent Router")
    st.write(
        "Classify an English online-banking support message into one of BANKING77's "
        "77 service intents."
    )

    if not LOCAL_MODEL_CONFIGURED:
        st.info(
            "The fine-tuned classifier is not available. Set `BANKING77_MODEL_ID` "
            "to a Hugging Face model repository containing the fine-tuned model, "
            "or run `python train.py` to create it locally. You can still try the "
            "pretrained retrieval baseline."
        )
        pipeline_options = ["Pretrained retrieval"]
    else:
        pipeline_options = [
            "Fine-tuned classifier",
            "Pretrained retrieval",
            "Compare both",
        ]

    selected_pipeline = st.selectbox("Prediction pipeline", pipeline_options)
    with st.expander("Advanced feature: voice-to-text"):
        st.write(
            "Record your enquiry to transcribe it into English before intent "
            "classification. You can also type a message below."
        )
        recorded_audio = st.audio_input(
            "Record a banking enquiry",
            sample_rate=16000,
        )

    query = st.text_area(
        "Customer message",
        placeholder="For example: I don't recognize a card payment on my account.",
        max_chars=2000,
    )
    if not st.button("Classify message", type="primary"):
        return
    message = query.strip()
    if recorded_audio is not None:
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

    if not message:
        st.warning("Enter a customer message or record an enquiry before classifying.")
        return

    if selected_pipeline == "Fine-tuned classifier":
        classifier = get_classifier(MODEL_REF)
        predictions = classifier.predict_many([message], top_k=3)[0]
        display_predictions(predictions, "Model probability")
    elif selected_pipeline == "Pretrained retrieval":
        retriever, _ = get_retriever()
        predictions = retriever.predict_many([message], top_k=3)[0]
        display_predictions(predictions, "Mean neighbor cosine similarity")
    else:
        classifier = get_classifier(MODEL_REF)
        retriever, _ = get_retriever()
        left, right = st.columns(2)
        with left:
            st.markdown("#### Fine-tuned classifier")
            display_predictions(
                classifier.predict_many([message], top_k=3)[0],
                "Model probability",
            )
        with right:
            st.markdown("#### Pretrained retrieval")
            display_predictions(
                retriever.predict_many([message], top_k=3)[0],
                "Mean neighbor cosine similarity",
            )


if __name__ == "__main__":
    main()
