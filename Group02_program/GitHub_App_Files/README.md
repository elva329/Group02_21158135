# BANKING77 Intent Routing

This project compares two Hugging Face-based approaches for routing English
online-banking queries to one of BANKING77's 77 intent categories:

1. A DistilBERT text classifier fine-tuned on BANKING77.
2. A pretrained sentence-embedding retriever that assigns the intent of the
   nearest training examples. It uses a balanced reference set and does not
   fine-tune a second model.

Sentiment is intentionally not treated as a project label: BANKING77 has intent
labels, not verified sentiment annotations.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dataset and pretrained model are downloaded from Hugging Face when first
used. Run these commands from this directory.

## Fine-tune the main model

```bash
python train.py
```

The script creates a stratified validation split from the official training
split, trains DistilBERT, and saves the model to
`Fine-tuned_Model_files/banking77-distilbert/`. It does not use the official test split for
training or model selection.

Optional training controls:

```bash
python train.py --epochs 3 --train-batch-size 16 --learning-rate 2e-5
```

## Compare both pipelines

```bash
python evaluate.py
```

This evaluates on the complete official test split and writes accuracy, macro
F1, inference time, and per-sample inference time to
`results/banking77_comparison.csv`. Retrieval-index construction time is
reported separately as `setup_seconds`. For a quick smoke test, use
`--max-test-samples 100`; use the full test set for reported results.

```bash
python evaluate.py --max-test-samples 100
```

## Run the Streamlit application

```bash
streamlit run app.py
```

Before training, the app offers the pretrained retrieval baseline. After
training, it also offers the fine-tuned classifier and a side-by-side
comparison.

## Deploy to Streamlit Community Cloud

1. Run `python train.py` and confirm the model works locally.
2. Create a **public model repository** on Hugging Face and upload the contents
   of `Fine-tuned_Model_files/banking77-distilbert/`. The repository must include the model
   config, weights, tokenizer files, and `label_names.json`. Keep a separate
   copy of the fine-tuned model for the course submission; generated model
   weights are excluded from Git because of their size.
3. Push the app source and `requirements.txt` to GitHub. Do not commit
   `.streamlit/secrets.toml`, model weights, or credentials.
4. In Streamlit Community Cloud, create an app from that GitHub repository and
   choose `app.py` as the main file.
5. In the app's **Advanced settings → Secrets**, set the root-level value
   below, replacing it with your public Hugging Face model repository ID:

   ```toml
   BANKING77_MODEL_ID = "your-hugging-face-username/banking77-distilbert"
   ```

   Community Cloud exposes root-level secrets as environment variables, which
   the app reads as `BANKING77_MODEL_ID`. Do not put an access token here for a
   public model. For local development, set the same environment variable in
   your shell if loading a model from the Hub.

6. Wait for the app to build, then test several held-out BANKING77 examples.
   Record the public app URL and model URL in the report, and capture app
   screenshots for the deployment section.

See the official [Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [secrets management guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
for the current deployment steps.

## Notes for reporting

- Keep the official test split untouched until final evaluation.
- Report model accuracy and runtime; macro F1 and a confusion matrix are useful
  supplementary analyses for the 77-class task.
- Retrieval scores are cosine similarities, not calibrated probabilities.
- Record package versions, training settings, test sample count, and the exact
  model repository/revision used in the final report.
- Check the dataset and model licenses before redistribution or deployment.
