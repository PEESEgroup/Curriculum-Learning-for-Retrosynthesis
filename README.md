# Rethinking Retrosynthesis: Curriculum Learning Reshapes Transformer-Based Small-Molecule Reaction Prediction

This repository provides a framework for retrosynthesis prediction using **Curriculum Learning (CL)** with difficulty-aware pacing. The implementation supports training on the USPTO-50K dataset and utilizes transformer-based models, including **ChemBERTa+DistilGPT2, ReactionT5v2, and BART**.

<p align="center">
  <img width="396" alt="image" src="https://github.com/PEESEgroup/Curriculum-Learning-for-Retrosynthesis/blob/main/TOC.png" />
</p>

## 📌 Features

- Train transformer-based models for **one-step retrosynthesis prediction**.
- Implements **curriculum learning (CL)** with difficulty-aware pacing based on:
  - Synthetic accessibility (SA) score
  - Ring count
  - Heavy atom count
- Supports multiple transformer architectures:
  - **ChemBERTa+DistilGPT2**
  - **ReactionT5v2**
  - **BART**
- Provides multiple curriculum pacing strategies:
  - **Linear**
  - **Logarithmic**
  - **Stepwise**
- Supports evaluation on:
  - **Random splits**
  - **Scaffold splits**
  - **Low-similarity test sets**
- Includes comprehensive evaluation metrics:
  - **Top-1 / Top-5 accuracy**
  - **SMILES validity**
  - **Scaffold-level generalization**
- Fully automated **difficulty score computation** from product SMILES using RDKit.
- Supports **bootstrap resampling** for robust metric uncertainty estimation.
- Built on **PyTorch**, **Hugging Face Transformers**, and **RDKit** for reproducibility.

## 📂 Repository Structure

```bash
retrosynthesis-curriculum-learning/
│
├── data/                                        # Dataset files used in the study
│   └── USPTO_50K.csv
│
├── evaluation/                                  # Model evaluation notebooks
│   ├── model-based-evaluations/                 # Evaluation scripts for each architecture
│   │   ├── BART/
│   │   │   └── Evaluation_of_curriculum_learning_for_retrosynthesis_(BART).ipynb
│   │   ├── ChemBERTa_DistilGPT2/
│   │   │   └── Evaluation_of_curriculum_learning_for_retrosynthesis_(ChemBERTa_distilgpt2).ipynb
│   │   └── ReactionT5v2/
│   │       └── Evaluation_of_curriculum_learning_for_retrosynthesis_(T5).ipynb
│   │
│   └── similarity-decay-analysis/               # Similarity decay analysis (as presented in the paper)
│       └── Evaluation_of_curriculum_learning_for_retrosynthesis_(ChemBERTa_+_DistilGPT2_similarity_decay_analysis).ipynb
│
├── training/                                    # Model training notebooks
│   ├── normal-training/                         # Non-scaffold training experiments
│   │   ├── BART/                                 # BART model experiments
│   │   │   ├── Curriculum_learning_for_retrosynthesis_(BART).ipynb
│   │   │   └── Curriculum_learning_for_retrosynthesis_(BART_baseline).ipynb
│   │   │
│   │   ├── ChemBERTa_DistilGPT2/                 # ChemBERTa + DistilGPT2 experiments
│   │   │   ├── Curriculum_learning_for_retrosynthesis_(ChemBERTa_distilgpt2).ipynb
│   │   │   └── Curriculum_learning_for_retrosynthesis_(ChemBERTa_distilgpt2_baseline).ipynb
│   │   │
│   │   └── ReactionT5v2/                         # ReactionT5v2 experiments
│   │       ├── Curriculum_learning_for_retrosynthesis_(T5).ipynb
│   │       └── Curriculum_learning_for_retrosynthesis_(T5_baseline).ipynb
│   │
│   └── scaffold-training/                        # Scaffold-split training experiments
│       └── Curriculum_learning_for_retrosynthesis_(ChemBERTa_distilgpt2_scaffold).ipynb
│
├── README.md                                    # Project documentation
```

## 🚀 Training & Evaluation Workflow

This repository includes full training and evaluation pipelines for retrosynthesis prediction using both:

- 🔹 **Baseline models** (random sampling)
- 🔹 **Curriculum Learning models** (difficulty-aware pacing)

All models are trained directly from the provided Jupyter notebooks.

---

### 1️⃣ Train Baseline Model (Random Sampling)

To train a **baseline model** without curriculum learning:

1. Navigate to the appropriate model folder under `notebooks/`.
2. Open and run the baseline notebook (filename has "baseline in its title next to the model name").

### 2️⃣ Train Curriculum Model (Difficulty-Aware Pacing)

To train a **curriculum model**:

1. Navigate to the appropriate model folder under `notebooks/`.
2. Open and run the curriculum notebook.

### 3️⃣ Evaluation

1. All evaluations (Top-1/Top-5 accuracy, SMILES validity) are performed directly inside each notebook.
2. Uncomment the respective cells and run to perform the corresponding evaluation.
3. Bootstrap resampling for metric uncertainty estimation is included.

## 🧪 Dataset

- Dataset: USPTO-50K <br/>
- Download directly from: [USPTO-50K](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/USPTO_50K.csv) <br/>
- Preprocessing and canonicalization are fully implemented inside the training notebooks.<br/>

## 📜 Citation
- If you find this repository useful, please cite our work:
```
@article{Sheshanarayana2025,
  author    = {R. Sheshanarayana and Fengqi You},
  title     = {Rethinking Retrosynthesis: Curriculum Learning Reshapes Transformer-Based Small-Molecule Reaction Prediction},
  journal   = {insert after publication},
  year      = {insert after publication},
  volume    = {insert after publication},
  pages     = {insert after publication},
  doi       = {insert after publication}
}
```

---

Feel free to contribute or raise issues! 🚀

