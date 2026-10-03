# ⚡ FinFET Drain Current Prediction

> **Academic Demonstration Prototype** — Machine learning–based prediction of FinFET drain current using physically-inspired synthetic data.

---

## 📖 Overview

This project demonstrates how machine learning can model the electrical behavior of **FinFET (Fin Field-Effect Transistor)** devices — the core building block of modern nanoscale CMOS technology (7 nm, 5 nm, 3 nm nodes and beyond).

A synthetic dataset is generated using simplified FinFET physics equations, three regression models are trained and compared, and an interactive **Streamlit dashboard** lets users predict drain current in real time by adjusting device and bias parameters.

> ⚠️ **Disclaimer**: The dataset is **synthetic and physically-inspired**, not derived from TCAD simulations or real silicon measurements. This project is intended for educational and demonstration purposes only.

---

## 🗂️ Project Structure

```
project 1/
├── generate_dataset.py   # Generates synthetic FinFET dataset (dataset.csv)
├── train_model.py        # Trains & evaluates ML models, saves model.pkl
├── app.py                # Streamlit interactive dashboard
├── requirements.txt      # Python dependencies
├── dataset.csv           # Generated synthetic dataset (6 000 samples)
├── model.pkl             # Saved best-performing model pipeline
├── model_metrics.json    # Training metrics & predictions for all models
└── docs/                 # Additional documentation (if any)
```

---

## 🔬 Physics Background

FinFETs use a 3D fin-shaped channel that allows the gate to wrap around three sides, providing superior electrostatic control compared to planar MOSFETs. The key device parameters modelled here are:

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Gate length | Lg | 7 – 50 | nm |
| Fin width | Wfin | 4 – 15 | nm |
| Fin height | Hfin | 20 – 60 | nm |
| Oxide thickness | Tox | 0.8 – 3 | nm |
| Threshold voltage | Vth | 0.2 – 0.5 | V |
| Gate-source voltage | VGS | 0 – 1.2 | V |
| Drain-source voltage | VDS | 0 – 1.0 | V |
| Temperature | Temp | 200 – 400 | K |

**Target**: Drain current **Id** (Amperes)

The physics model captures:
- **Above-threshold current** — long-channel MOSFET model with effective width (`Weff = Wfin + 2·Hfin`)
- **Sub-threshold leakage** — exponential dependence on overdrive voltage
- **Short-channel effects (SCE)** — exponential boost for small gate lengths
- **Temperature-dependent mobility** — `μ ∝ (T/T₀)⁻¹·⁵`
- **Smooth transition** — sigmoid blending between above/sub-threshold regimes
- **Realistic noise** — ±0.5% Gaussian noise

---

## 🤖 Machine Learning Models

Three regression models are trained and compared using an 80/20 train-test split:

| Model | Description |
|-------|-------------|
| **Linear Regression** | Ridge regression (α = 1.0) with StandardScaler |
| **Random Forest** | 200 trees, max depth 20, all CPU cores |
| **Gradient Boosting** | 300 estimators, depth 6, learning rate 0.08 |

The best model (by test R²) is automatically selected and saved as `model.pkl`. Gradient Boosting typically achieves the highest accuracy.

**Metrics reported**: R² (train & test), MAE, RMSE

---

## 🚀 Getting Started

### 1. Clone / download the project

```bash
git clone <repo-url>
cd "project 1"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset

```bash
python generate_dataset.py
```

This creates `dataset.csv` with 6 000 synthetic FinFET samples.

### 4. Train the models

```bash
python train_model.py
```

This trains all three models, prints a comparison table, and saves:
- `model.pkl` — the best model pipeline
- `model_metrics.json` — detailed metrics for the Streamlit dashboard

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Dashboard Features

The Streamlit dashboard (`app.py`) provides:

- **Real-time prediction** — adjust device and bias sliders to instantly predict Id
- **Model comparison** — R², MAE, and RMSE for all three trained models
- **Feature importance** — Random Forest importances visualized with Plotly
- **Scatter plots** — Predicted vs. actual Id for each model
- **Dataset explorer** — interactive inspection of the generated dataset
- Dark-themed, premium UI with GitHub-style color palette

---

## 📦 Dependencies

| Package | Version |
|---------|---------|
| streamlit | ≥ 1.40.0 |
| pandas | ≥ 1.5.0 |
| numpy | ≥ 1.23.0 |
| scikit-learn | ≥ 1.2.0 |
| plotly | ≥ 5.15.0 |
| joblib | ≥ 1.2.0 |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🧪 Reproducibility

- Random seed `42` is used throughout (`numpy`, `scikit-learn`) for fully reproducible results.
- Re-running `generate_dataset.py` always produces the same dataset.
- Re-running `train_model.py` always produces the same model weights.

---

## 📝 Limitations & Future Work

- **Synthetic data only** — results do not reflect real silicon or TCAD accuracy.
- **Simplified physics** — no quantum confinement, ballistic transport, or interface traps.
- **Potential improvements**:
  - Replace synthetic data with TCAD-simulated (e.g., Sentaurus) or measured data
  - Add deep learning models (MLP, Graph Neural Networks for device structures)
  - Extend to multi-finger / multi-fin configurations
  - Include variability / Monte Carlo analysis

---

## 👤 Author

**Mohit Roscen**
Academic Prototype — FinFET ML Prediction
2026

---

## 📄 License

This project is for academic and educational use only. No warranty is provided regarding physical accuracy.
