# BANKING77 Intent Routing Project

This project classifies English online-banking support messages into one of
BANKING77's 77 service intents. It includes a fine-tuned DistilBERT classifier,
a pretrained embedding-retrieval baseline, a Streamlit app, training and
evaluation scripts, and Colab-ready notebooks.

## Project layout

- `Group02_program/GitHub_App_Files/` — Streamlit app, shared inference code,
  training/evaluation scripts, dependencies, and tests.
- `Group02_program/Python_notebooks/` — notebooks for fine-tuning and
  evaluating the pipelines.
- `Group02_Dataset_files/` — prepared train/validation/test CSV files and the
  submitted fine-tuned model.
- `Group02_documentation/` — report and experiment results.
- `Group02_presentation/` — presentation materials.

## Requirements

- Python 3.11 or newer.
- Internet access the first time you run the app, train, or evaluate: Hugging
  Face downloads the BANKING77 dataset and the pretrained embedding model.
- A GPU is optional. Training is faster with one; inference also supports CPU.

## Install dependencies

Run these commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Group02_program/GitHub_App_Files/requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run the Streamlit app

From the project root, activate the virtual environment and launch Streamlit:

```bash
cd Group02_program/GitHub_App_Files
streamlit run app.py
```

The application automatically locates the fine-tuned model included in
`Group02_Dataset_files/Fine-tuned_Model_files/banking77-distilbert`. You can
also override this path or point to a public Hugging Face repository using
the `BANKING77_MODEL_ID` environment variable:

```bash
export BANKING77_MODEL_ID="username/banking77-distilbert"
```

Open the local URL printed by Streamlit (normally
`http://localhost:8501`). Enter a banking-support message, choose a pipeline,
and select **Classify message**. The app offers:

- **Fine-tuned classifier** — predicts the intent with DistilBERT.
- **Pretrained retrieval** — finds similar training examples with MiniLM
  embeddings.
- **Compare both** — displays both predictions side by side.

If no local model or environment variable is configured, the app will offer
the pretrained retrieval baseline.

To stop the app, press `Ctrl+C` in the terminal.

## Fine-tune DistilBERT

Run the training script from the app source directory:

```bash
cd Group02_program/GitHub_App_Files
python train.py
```

By default, training uses the official BANKING77 training split, creates a
stratified validation split, and saves the model to
`Fine-tuned_Model_files/banking77-distilbert/` inside the app source directory.
It does not use the official test split for training or checkpoint selection.

Optional training parameters:

```bash
python train.py --epochs 3 --train-batch-size 16 --learning-rate 2e-5
```

## Evaluate the pipelines

From `Group02_program/GitHub_App_Files/`, evaluate against the full official test split
(the script automatically detects the model in `Group02_Dataset_files`):

```bash
python evaluate.py
```

For a quick smoke test, limit evaluation to 100 examples:

```bash
python evaluate.py --max-test-samples 100
```

The script reports accuracy, macro F1, and inference runtime for both
pipelines. It writes CSV and Excel results under `results/` in the app source
directory. Omit `--model-dir` if you trained a model using the default output
directory above.

## Run the notebook

The project notebook is located in `Group02_program/Python_notebooks/`:

- **Master Notebook (`Banking77_Intent_Routing.ipynb`)**:
  A single, self-contained notebook covering the entire deep learning workflow
  end-to-end:
  1. Pinned dependency installation and deterministic seed configuration.
  2. Dataset loading and stratified 90/10 train/validation splitting (leaving the official test split untouched).
  3. Fine-tuning DistilBERT sequence classification (Pipeline 1) with checkpoint selection on validation accuracy.
  4. Pretrained sentence-embedding retrieval using MiniLM feature extraction (Pipeline 2).
  5. Comprehensive test-set evaluation comparing accuracy and inference runtime across both pipelines.
  6. Confusion matrix analysis for top intent misclassifications.
  7. Results export to CSV and Excel workbooks.
  8. Interactive live query demo.

To run the notebook in Google Colab or Jupyter, open `Group02_program/Python_notebooks/Banking77_Intent_Routing.ipynb`
and select **Runtime → Restart session and run all**. The notebook comes with all outputs pre-rendered
matching the documented experimental results.

## Run the unit tests

From the project root, with the virtual environment active:

```bash
cd Group02_program/GitHub_App_Files
python -m unittest discover -s tests -q
```

## Deployment note

The local app can use the included model by setting `BANKING77_MODEL_ID` to
its local directory as shown above. A hosted deployment that loads the model
from Hugging Face instead needs `BANKING77_MODEL_ID` set to the public Hugging
Face model repository ID in the deployment's environment/secrets. The app
requirements support the Python 3.14 runtime currently used by Streamlit
Community Cloud and prefer CPU-only PyTorch wheels for its hosted CPU runtime.
The PyTorch wheel page is configured as a find-links source, not an extra
package index, so it does not interfere with resolving other dependencies.
