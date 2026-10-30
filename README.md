# Speech + Text Depression Detection with Supervised Contrastive Fusion and Latent-Space Oversampling

Research code and manuscript for a lightweight audio + text pipeline on the free **Androids Corpus** (Italian clinical interviews, 116 speakers: 64 patients, 52 controls).

> **Research prototype only. Not a diagnostic tool and not for clinical use.**

## What it does

1. **Features (frozen):** 20 s windows. Audio: XLSR-53 hidden states, mean + std pooled (2048-d). Text: Whisper (`small`, via faster-whisper) transcripts embedded with `paraphrase-multilingual-mpnet-base-v2` (768-d).
2. **Fusion network:** per-modality MLPs (256-d) fused into a shared latent space `z`, trained with class-weighted cross-entropy + supervised contrastive loss (`lambda = 0.5`, `tau = 0.1`).
3. **Latent-space oversampling:** frozen encoder, SMOTE on `z` (training windows only), logistic-regression classifier.
4. **Evaluation:** speaker-grouped stratified 5-fold CV (no speaker in both train and test), speaker-level metrics, bootstrap CIs over speakers, overlap diagnostics, leakage study, confounder-only baselines, and a controlled imbalance study.
5. **Demo:** Gradio microphone/upload interface (audio + text only; no video modality).

## Run it

Open `notebooks/androids_depression_pipeline.ipynb` in Google Colab, select a **GPU** runtime, and run the single code cell. It downloads the corpus, extracts embeddings (resumable), runs every experiment, prints all tables, saves CSV tables and figures to `/content/androids_results`, downloads them as a zip, and launches the demo. Knobs at the top of the cell: `EPOCHS`, `LAM`, `REPEATS`, `RUN_IMB`, `RATIOS`.

Local install: `pip install -r requirements.txt` (a CUDA GPU is strongly recommended).

## Selected results (imbalance study)

Speaker-level macro-F1 when patients are under-sampled in the training folds (patient : control ratio; test folds untouched; single 5-fold split):

| Method | 1:2 | 1:4 | 1:8 |
|---|---|---|---|
| MLP (CE), no correction | 0.684 | 0.598 | 0.472 |
| MLP (CE) + class weights | 0.712 | 0.654 | 0.581 |
| **SupCon + latent SMOTE (proposed)** | **0.785** | **0.742** | **0.704** |

![Imbalance study](figures/fig_imbalance.png)

UMAP of held-out windows (first test fold): raw embeddings, cross-entropy-only latent space, and the proposed contrastive latent space.

![UMAP](figures/fig_umap.png)

The near-complete separation in the right panel should be read with care: UMAP can exaggerate separation, the figure shows one fold, and on a small single-site corpus recording conditions can act as shortcuts. See the paper's Discussion and Limitations.

## Repository layout

```
notebooks/androids_depression_pipeline.ipynb   full pipeline (one Colab cell)
paper/ml_paper/                                Elsevier (elsarticle) manuscript source, PDF, highlights, figures
figures/                                       figures used in the paper
requirements.txt
```

## Data

The **Androids Corpus** is not included and must not be redistributed. It is free for academic, non-commercial use; see the corpus repository: <https://github.com/androidscorpus/data>. The notebook downloads it automatically.

## Reference for the corpus

F. Tao, A. Esposito, A. Vinciarelli. *The Androids Corpus: A new publicly available benchmark for speech based depression detection.* Interspeech 2023.

## License

MIT (code). The dataset and pretrained models keep their own licences and terms.
