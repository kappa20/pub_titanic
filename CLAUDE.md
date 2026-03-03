# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Activate the conda environment first
conda activate python_3_11_13

# Install dependencies (if needed)
pip install -r requirements.txt

# Start the Streamlit app (available at http://localhost:8501)
streamlit run app.py

# In GitHub Codespaces (with CORS disabled for devcontainer):
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

## Important Constraints

- **Python 3.11.13** — The model is trained with Python 3.11.13 to match Streamlit Cloud's runtime. Use `conda activate python_3_11_13` to run the app locally.
- **scikit-learn is pinned to 1.3.2** in `requirements.txt` for Streamlit Cloud compatibility. Do not upgrade it without retraining `models/best_model.pkl` with the new version.
- The model file is a `GridSearchCV` object — `load_model()` in `src/model_utils.py` extracts the best estimator's pipeline components from it.

## Architecture

The app follows a layered structure:

- **`app.py`** — Streamlit entry point. Orchestrates page flow, calls `src/` modules, and handles session state.
- **`config/app_config.py`** — Single source of truth for file paths, color scheme, feature metadata, validation ranges, and default values.
- **`src/feature_engineering.py`** — Replicates training-time feature derivation at prediction time. Any new features added here must match those used when `models/best_model.pkl` was trained.
- **`src/model_utils.py`** — Loads the model, runs predictions, extracts feature importances from LR coefficients, and generates plain-English explanations.
- **`src/visualization.py`** — Plotly charts (gauge, bar chart, profile card, comparison metrics).
- **`src/ui_components.py`** — All Streamlit UI components (input form, results display, model info panel).

### ML Pipeline

The model is a scikit-learn `Pipeline` containing:
1. `ColumnTransformer` — `StandardScaler` on numerics, `OneHotEncoder` on categoricals
2. `LogisticRegression`

Engineered features (created in `feature_engineering.py` and expected by the pipeline):
- `FamilySize` = SibSp + Parch + 1
- `IsAlone` = 1 if FamilySize == 1
- `Title` = extracted from Name (Mr/Mrs/Miss/Master/Rare)
- `Deck` = first letter of Cabin (or 'Unknown')
- `HasCabin` = binary indicator
- `FarePerPerson` = Fare / FamilySize

### Data

Training data lives in `titanic/train.csv` (891 samples). The notebook `titanic_ml_project.ipynb` documents the full model development and reports ~82.7% accuracy / 0.868 AUC-ROC on the validation set.

## Streamlit Caching

- `@st.cache_resource` is used for model loading (persists across reruns)
- `@st.cache_data` is used for data loading

Invalidate caches during development with the Streamlit menu or by restarting the server.
