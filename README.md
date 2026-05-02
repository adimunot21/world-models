# World Models: From Dreams to JEPA

A production-quality implementation comparing two paradigms of world models — **generative** (Ha & Schmidhuber 2018) and **joint-embedding predictive** (JEPA / LeWorldModel 2026) — on the CarRacing-v3 reinforcement learning environment.

## What This Project Does

This project builds and compares two approaches to learning an internal simulator ("world model") of an environment:

**Paradigm 1 — Generative (VAE + MDN-RNN + Controller):**
Compress frames to latent vectors with a VAE, predict future latent states with a mixture density network, and train a controller to drive using evolutionary optimization (CMA-ES). The agent can even train inside the model's own "dreams" (imagination-based training).

**Paradigm 2 — JEPA (Encoder + Predictor + SIGReg):**
Encode frames to embeddings without any decoder. Predict future embeddings directly. Regularize the embedding space with SIGReg to be Gaussian-distributed. No pixel reconstruction at any point.

The project also includes a deep study of [LeWorldModel](https://github.com/lucas-maes/le-wm), the state-of-the-art JEPA-based world model, with pretrained weights and analysis.

## Key Results

| Experiment | Result |
|-----------|--------|
| VAE reconstruction quality | Captures road geometry, car position, scene layout |
| MDN-RNN prediction | Loss: +10.67 → -43.26 over 50 epochs (accurate dynamics) |
| Real-trained controller | Best episode: 643.9, mean: 180.4 ± 212.0 |
| Dream-trained controller | Mean: -80.5 (heuristic reward doesn't transfer) |
| JEPA embedding structure | Isotropic Gaussian (SIGReg), 2.3% PCA vs VAE's 39.7% |
| Nonlinear steering probe | JEPA: 0.14 R² vs VAE: 0.07 R² |

**Key finding:** Dream training fails without an accurate reward model — the controller exploits the heuristic reward instead of learning to drive. JEPA's goal-conditioned planning (MSE to goal embedding) sidesteps this problem entirely, which is why it's the dominant approach in modern world model research.

## Architecture

```
┌────────────────────────────┬──────────────────────────────────┐
│  GENERATIVE (Ha & Schmid.) │  JEPA (LeWorldModel-inspired)    │
│                            │                                  │
│  Frame → VAE → z (32d)     │  Frame → CNN → emb (192d)        │
│  (z,a)  → MDN-RNN → p(z') │  (emb,a) → MLP → emb' (192d)    │
│  [z,h]  → Controller → a  │  SIGReg enforces Gaussian emb    │
│  Decoder reconstructs      │  NO decoder. NO reconstruction.  │
│  KL pushes z toward N(0,1) │  Planning: MSE to goal embedding │
└────────────────────────────┴──────────────────────────────────┘
```

## Project Structure

```
world-models/
├── src/
│   ├── data/           # Data collection, loading, inspection
│   ├── models/         # VAE, MDN-RNN, Controller, JEPA
│   ├── training/       # Training loops for all models
│   ├── evaluation/     # Agent, dream env, paradigm comparison
│   ├── visualization/  # Reconstruction, latent space plots
│   └── utils/          # Config, logging, reproducibility
├── lewm/               # LeWorldModel clone + analysis
├── scripts/            # Demo and entry-point scripts
├── config/             # YAML configs for all models
├── tests/              # Unit and integration tests
├── notebooks/          # Analysis notebooks
├── course/             # Multi-chapter educational course
└── results/            # Generated plots and videos
```

## Setup

**Requirements:** Python 3.10, CUDA-capable GPU (4GB+ VRAM), ~10GB disk space.

```bash
# Clone
git clone git@github.com:adimunot21/world-models.git
cd world-models

# Create environment
conda create -n world-models python=3.10 -y
conda activate world-models

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "numpy<2"
pip install -r requirements.txt
```

## Quick Start

```bash
# Collect data (500 episodes, ~2-3 hours)
python src/data/collect.py --num_rollouts 500

# Train VAE (~25 min)
python src/training/train_vae.py

# Encode dataset through VAE (~3 min)
python src/training/encode_dataset.py

# Train MDN-RNN (~50 min)
python src/training/train_mdn_rnn.py

# Train controller with CMA-ES (~2-4 hours)
python src/training/train_controller.py --pop_size 16 --num_rollouts 1 --max_steps 500

# Train JEPA for comparison (~9 hours)
python src/training/train_jepa.py

# Run comparison analysis
python src/evaluation/compare.py

# Run demo
python scripts/run_demo.py
```

## Technology Stack

| Library | Purpose |
|---------|---------|
| PyTorch | Core ML framework |
| Gymnasium | CarRacing-v3 environment |
| CMA (pycma) | Evolutionary optimization for controller |
| stable-worldmodel | LeWorldModel environments and data |
| scikit-learn | Linear probing, PCA analysis |
| matplotlib + seaborn | Visualization |
| TensorBoard | Training monitoring |

## Course

A multi-chapter course explaining world models from first principles is in `course/`. Each chapter covers the theory, design decisions, implementation, and results of one project component.

| Chapter | Topic |
|---------|-------|
| 00 | Introduction — why world models matter |
| 01 | Environment and Data — RL basics, CarRacing, data collection |
| 02 | Variational Autoencoders — latent spaces, ELBO, reparameterization |
| 03 | Mixture Density Networks — multimodal prediction, MDN-RNN |
| 04 | Dreaming and Imagination — dream training, where it fails |
| 05 | Evolutionary Control — CMA-ES, why evolution works for RL |
| 06 | The JEPA Paradigm — why pixel prediction is wasteful |
| 07 | LeWorldModel — SIGReg, end-to-end JEPA from pixels |
| 08 | Bridging Paradigms — VAE vs JEPA on the same data |
| 09 | V-JEPA 2 and Scale — internet-scale pretraining |
| 10 | Integration and Retrospective — lessons learned, future directions |

## Hardware

Developed and tested on:
- NVIDIA GeForce GTX 1650 (4GB VRAM)
- Intel i7-9750H, 32GB RAM
- Ubuntu 24.04

All models train locally on a single consumer GPU. V-JEPA 2 analysis (Phase 8) uses Google Colab T4.

## License

MIT

## Acknowledgments

- [Ha & Schmidhuber 2018](https://worldmodels.github.io/) — the foundational world model architecture
- [LeWorldModel (Maes et al. 2026)](https://github.com/lucas-maes/le-wm) — state-of-the-art JEPA world model
- [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) — environments and data infrastructure