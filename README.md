# LoRA Fine-Tuning for Code-Mixed Sentiment Classification

This repository contains a clean implementation of **LoRA (Low-Rank Adaptation)** fine-tuning using HuggingFace PEFT for **code-mixed sentiment classification** in Tamil–English and Hindi–English.

The goal is to show how **parameter-efficient fine-tuning** can retain almost all of a multilingual teacher model’s performance while training **<1% of the parameters**.

---

## 📁 Repository Structure

- `notebooks/`
  - `LoRA_trainer.ipynb` – main notebook for LoRA fine-tuning and evaluation
  - `ResultSummary.ipynb` – (optional) analysis/aggregation of results
- `run_colab.ipynb` – one-click Colab runner that:
  - Clones this repo into `/content/LoRA-finetuning-Codemix`
  - Installs dependencies from `requirements.txt`
  - Verifies key imports (`torch`, `transformers`, `peft`)
- `requirements.txt` – dependencies for Colab/local
- `.gitignore` – ignores notebook checkpoints, caches, and other junk
- `LICENSE` – MIT License (open and reusable)

---

## 🚀 How to Run (Colab)

1. **Open the Colab runner**

   Open `run_colab.ipynb` in Colab (via GitHub → “Open in Colab”).

2. **Run all cells in `run_colab.ipynb`**

   This will:

   - Clone the repo to `/content/LoRA-finetuning-Codemix`
   - Install all requirements from `requirements.txt`
   - Check imports for `torch`, `transformers`, and `peft`

3. **Open the main LoRA notebook**

   In the Colab file browser, open:

   - `notebooks/LoRA_trainer.ipynb`

4. **Run the LoRA notebook**

   - The first cell sets deterministic seeding (`project_seed(42)`).
   - Follow the cells to:
     - Load the code-mixed dataset (Tamil–English / Hindi–English)
     - Load the multilingual DistilBERT teacher
     - Attach LoRA adapters and fine-tune
     - Evaluate and log metrics, saving a small JSON summary

> If you store your dataset on Google Drive, adjust the `DRIVE_BASE` / path variables in the notebook to point to your CSV splits (e.g., `train/val/test`).

---

## 🧠 Methodology (High-Level)

- **Base model**: `distilbert-base-multilingual-cased`
- **Task**: Binary sentiment classification on:
  - Tamil–English code-mixed dataset
  - Hindi–English code-mixed dataset
- **Approach**:
  - Start from a **fully fine-tuned teacher** (same architecture as KD project)
  - Freeze all base model parameters
  - Attach LoRA adapters to selected Transformer modules
  - Only train the low-rank adapter parameters

We explore the effect of three hyperparameters:

- **Rank (`r`)**
- **Scaling factor (`α`)**
- **Dropout**

and study how they impact accuracy, macro-F1, and parameter count.

---

## 📊 Results

Using a multilingual DistilBERT teacher and LoRA on top:

- **Teacher parameters** (full model): ~136M  
- **LoRA trainable parameters** (best config): ~0.6M  
- **Trainable parameter reduction**: **>99.5%**

### Tamil–English (Code-Mixed)

Best configuration:

- **LoRA config**: `r = 4`, `α = 64`, `dropout = 0.05`
- **Teacher macro-F1**: **0.752**
- **LoRA macro-F1**: **0.752** (essentially unchanged)

| Model                    | Macro-F1 | Comment                  |
|--------------------------|----------|--------------------------|
| Teacher (full fine-tune) | 0.752    | Multilingual DistilBERT  |
| LoRA (r=4, α=64, p=0.05) | 0.752    | ≈99.5% fewer trainable params |

### Hindi–English (Code-Mixed)

Same LoRA configuration:

- **Teacher macro-F1**: **0.885**
- **LoRA macro-F1**: **0.889** (slight improvement)

| Model                    | Macro-F1 | Comment                                   |
|--------------------------|----------|-------------------------------------------|
| Teacher (full fine-tune) | 0.885    | Multilingual DistilBERT                  |
| LoRA (r=4, α=64, p=0.05) | 0.889    | Slightly better, ≈99.5% fewer params     |

### Key Takeaways

- LoRA preserves **>99%** of teacher performance on both languages.
- On Hindi–English, LoRA slightly **outperforms** the teacher.
- Trainable parameters drop from ~136M to ~0.6M, demonstrating **extreme parameter efficiency**.
- Behaviour is consistent across **two linguistically distinct code-mixed corpora**, highlighting strong multilingual generalization.

---

## 🔁 Reproducibility

All experiments are implemented in notebooks and use a simple, self-contained seeding utility:

```python
import os, random
import numpy as np, torch

def project_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

project_seed(42)
