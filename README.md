# Steam Market Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://playtics-analytics.streamlit.app/)

An empirically driven, machine-learning-backed analytical platform designed to help indie developers and AA publishers enter the Steam marketplace with data-backed strategies.

The platform provides a complete **Cyberpunk UI** aesthetic, engineered for high-performance interactivity without browser lockups, seamlessly analyzing over 126,000 commercially released titles from 2010–2025.

---

## Key Features

### 1. Gamers Hub (Game Discovery)
- **Deep Exploration:** Filter and explore Steam's massive library with dynamic sliders for Price, Quality, and Playtime.
- **Similar Games Engine:** Find direct competitors and related games using a 9-feature standardized Cosine Similarity algorithm.
- **Radar Charts & Distributions:** Visualize exactly how a game compares to the rest of the action genre in terms of CCU, reviews, playtime, languages, and pricing.

### 2. Developer Hub (Strategic Insights)
- **AI-Powered Diagnostics:** Leverage tree-based machine learning (HistGradientBoosting) to analyze market dynamics and predict fair pricing.
- **SHAP Feature Importance:** Transparent ML diagnostics computed *live* to explain exactly why a game's price is predicted higher or lower (e.g., how much extra value a multiplayer tag or additional language support adds).
- **Competitor Benchmarking:** Analyze Top-5 genre competitors to see exactly where your planned title sits in the market hierarchy.

### 3. Data & Methodology (EDA Notebook)
- **Interactive Jupyter-Style Dashboard:** Directly view the 15-question Exploratory Data Analysis (EDA) process inside the Streamlit app.
- **PDF Extraction Engine:** Uses `PyMuPDF` and `Pillow` to dynamically crop, dark-theme invert, and render high-resolution executive charts directly from the source PDF report.
- **Strategic Takeaways:** Caches and displays "Golden Rules" and actionable stakeholder insights derived directly from historical data.

---

## Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/) + Custom CSS (`JetBrains Mono`, `IBM Plex Sans`, Cyberpunk palette: `#0a0a14`, `#00f5ff`, `#ff0066`).
- **Data Visualizations:** [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) & [SHAP](https://shap.readthedocs.io/)
- **PDF & Image Processing:** [PyMuPDF](https://pymupdf.readthedocs.io/) & [Pillow](https://python-pillow.org/)
- **Data Processing:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)

---

## Repository Structure

```text
steam-market-intelligence/
├── app.py                     # Streamlit entry point & sidebar routing
├── app/
│   └── pages/                 # Core Dashboard Views
│       ├── gamers.py          # Gamers Hub UI & Logic
│       ├── developers.py      # Developer Hub UI & Logic
│       └── methodology.py     # EDA Notebook UI & Logic
├── src/                       # Backend Logic & Configuration
│   ├── config.py              # Centralized constants and UI tokens
│   ├── data_loader.py         # Memory-cached data ingestion
│   ├── feature_engineering.py # Data transforms and pipelines
│   ├── model_loader.py        # ML model serving
│   ├── similarity.py          # Nearest-neighbor algorithms
│   └── validation.py          # Validation utilities
├── models/                    # Trained .pkl models (Sweetspot, Review Score, Tiers, etc.)
├── data/                      # Dataset location (e.g., steam_games_cleaned.csv)
├── assets/                    # Cached UI elements (e.g., dynamically extracted PDF charts)
├── notebooks/                 # Original EDA Jupyter notebooks
└── Steam_Market_Executive_Report.pdf  # Source of truth for EDA Notebook
```

---

## Installation & Usage

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/Istiyak-Ishan/playlytics.git
cd playlytics
```

### Windows (PowerShell)
```powershell
# Create a virtual environment
python -m venv venv

# Activate the environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

### Linux / macOS
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

*Note: If the Streamlit application returns an error regarding missing PDF charts, ensure that `Steam_Market_Executive_Report.pdf` is present in the root directory.*
