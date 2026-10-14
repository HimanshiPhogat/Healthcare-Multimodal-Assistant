# Multimodal Healthcare Assistant

A decision-support prototype that combines a **chest X-ray** and a **free-text clinical note** to estimate the likelihood of a respiratory infection (pneumonia-type presentation), explain the result, and surface relevant reference material.

> **Disclaimer:** This is a learning/portfolio project. It is not a medical device, has not been clinically validated, and must not be used for diagnosis or treatment decisions.

## What it does

```
 chest X-ray ──► ResNet-18 (fine-tuned) ──► P(image) ──┐
                                                       ├─► weighted late fusion ─► likely / unlikely / needs_review
 clinical note ─► TF-IDF + Logistic Regression ─► P(text) ─┘            │
                                                                        └─► TF-IDF retrieval over a small
                                                                            curated knowledge base (references)
```

Design decisions worth knowing about:

- **Late fusion, not a joint model.** The two modalities come from different datasets (no paired X-ray/note data is freely available), so each model is trained separately and combined at the probability level. This also keeps either input optional at inference time.
- **Abstention.** If the modalities disagree about which side of the threshold they fall on, or the fused score lands in an uncertain band, the output is `needs_review` instead of a forced yes/no.
- **Explainability.** Grad-CAM heatmaps for the image model, and the highest-contributing n-grams for the text model.
- **Graceful degradation.** The API and UI work with only the text model, only the image model, or both. PyTorch is imported lazily.

## Project layout

```
configs/config.yaml        all hyperparameters and paths
src/mmhealth/
  data/                    image dataset + synthetic note generator
  models/                  ResNet-18 head, text pipeline, Grad-CAM
  rag/                     markdown knowledge base + TF-IDF retriever
  fusion.py                late fusion + abstention logic
  pipeline.py              HealthAssistant: the end-to-end entry point
  train_image.py           image training/eval script
  train_text.py            text training/eval script
  api/main.py              FastAPI service
app/streamlit_app.py       demo UI
scripts/gradcam_demo.py    save a Grad-CAM overlay for one image
tests/                     pytest suite
```

## Setup

```bash
git clone <this-repo>
cd healthcare-multimodal-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src            # on Windows PowerShell: $env:PYTHONPATH="src"
```

## Data

**Images.** Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle and extract it to `data/chest_xray/` so that it contains `train/`, `val/`, `test/`, each with `NORMAL/` and `PNEUMONIA/` subfolders.

**Text.** Real clinical notes need credentialed access (for example MIMIC), so this repo generates **synthetic template-based notes** (`data/text_data.py`). That makes the pipeline reproducible, but it is a stand-in: the text model scores near-perfect on it because the templates are easy to separate. Treat those numbers as a sanity check of the code, not as evidence of clinical performance. Swapping in real data only requires replacing `generate_notes()` with a loader that returns `note` and `label` columns.

## Train

```bash
python -m mmhealth.train_text      # seconds, CPU
python -m mmhealth.train_image     # needs the Kaggle data; GPU recommended
```

Metrics are written to `outputs/`. The image training script uses transfer learning from ImageNet weights, a class-weighted loss (the dataset is imbalanced), a frozen-backbone warm-up for the first epochs, then full fine-tuning with a cosine schedule.

## Run

```bash
# API
uvicorn mmhealth.api.main:app --port 8000
curl -X POST localhost:8000/analyze \
  -F "note=54 y/o male, productive cough for 6 days, fever of 38.9 C, shortness of breath" \
  -F "image=@path/to/xray.jpeg"

# Demo UI
streamlit run app/streamlit_app.py

# Docker
docker build -t mmhealth . && docker run -p 8000:8000 mmhealth
```

Example response (abridged):

```json
{
  "decision": "likely",
  "fused_score": 0.93,
  "modalities_used": ["image", "text"],
  "image_probability": 0.91,
  "text_probability": 0.97,
  "key_terms": ["productive cough", "fever", "shortness breath"],
  "references": [{"source": "red_flags.md", "title": "Red flags requiring urgent assessment", "score": 0.31}],
  "disclaimer": "Research prototype. ..."
}
```

## Tests

```bash
pytest -q
```

Covers the synthetic data generator, text model (training, persistence, signal), fusion logic and abstention, retrieval, Grad-CAM/image model shapes (skipped if PyTorch is missing), and the end-to-end pipeline.

## Results

Fill this in after you train on your own machine, with the exact numbers from `outputs/image_metrics.json`:

| Model | Dataset | Metric | Value |
|-------|---------|--------|-------|
| Image (ResNet-18) | Kaggle chest X-ray, test split | ROC-AUC / recall (pneumonia) | _your result_ |
| Text (TF-IDF + LR) | Synthetic notes | ROC-AUC | ~1.0 (templates are easily separable) |

## Limitations and next steps

- The Kaggle test split is small and comes from a single paediatric source, so results will not generalize to other hospitals or age groups. External validation is the first thing a real project would need.
- The fusion weights are set by hand in the config. With paired data they could be learned (stacking) and calibrated (Platt scaling / isotonic regression).
- Retrieval is keyword-based over five short documents. A dense-embedding retriever and a larger, properly sourced knowledge base would be the natural upgrade, followed by an LLM that answers *only* from retrieved passages.
- No fairness or subgroup analysis has been done.

## License

MIT
