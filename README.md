<div align="center">

# 🛡️ RoadZen

### AI-Powered Traffic Safety Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blueviolet)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Real-time accident risk prediction · Explainable AI insights · Heatmap monitoring · Trauma center alerts

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Core Theme & Vision](#-core-theme--vision)
- [Live Features](#-live-features)
- [System Architecture](#-system-architecture)
- [How It Works — Business Logic](#-how-it-works--business-logic)
- [ML Model & Training Pipeline](#-ml-model--training-pipeline)
- [API Reference](#-api-reference)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [Dataset](#-dataset)
- [Author](#-author)

---

## 🚀 Overview

**RoadZen** is a full-stack, AI-powered road safety platform built to predict, explain, and mitigate traffic accident risk in real time. It combines a trained **XGBoost** machine learning model with **SHAP-based explainability**, an interactive **accident heatmap**, a multi-chart **analytics dashboard**, an **AI safety chatbot**, and an **emergency alert system** — all accessible through a clean, responsive web UI.

The platform ingests historical Indian road accident data, trains a multi-class severity classifier, and serves predictions over a **FastAPI** REST API. The frontend communicates with this API to deliver instant risk scores, AI-driven explanations, and situational awareness to drivers, fleet managers, or road safety analysts.

---

## 🎯 Core Theme & Vision

| Theme | Description |
|---|---|
| **Predictive Safety** | Know your accident severity risk *before* you drive — powered by ML |
| **Explainable AI** | SHAP values make every prediction transparent and understandable |
| **Situational Awareness** | Live heatmaps reveal accident hotspots across regions |
| **Emergency Readiness** | One-click alert dispatches to the nearest trauma centers |
| **Data-Driven Insights** | Six analytics charts surface patterns across time, weather, and vehicle type |

The guiding principle: **move from reactive accident response to proactive risk prevention.**

---

## ✨ Live Features

### 🎯 Risk Prediction Engine
- Input driving parameters (hour, age, weather, vehicle type, etc.)
- XGBoost model returns a **three-class severity prediction**: Minor · Moderate · Severe
- Visual gauge with confidence score and probability breakdown per class

### 🔍 Explainable AI (SHAP)
- Per-prediction SHAP waterfall explaining *which features drove* the result
- Top-5 contributing factors with direction (increases / decreases risk)
- Built on `shap.TreeExplainer` for fast, exact Shapley values

### 🗺️ Accident Heatmap
- Leaflet.js map with a `leaflet-heat` overlay showing accident density
- Colour-coded intensity: 🔴 High Risk → 🟠 Medium → 🟢 Low
- 10 real Delhi-area **trauma center** markers with type, speciality, phone & bed count
- Toggle heatmap / hospital layers independently

### 📊 Analytics Dashboard (6 charts)
| Chart | What it shows |
|---|---|
| Hourly Risk Distribution | Average severity by hour of day (peak danger windows) |
| Weather Impact Analysis | Severity increase per weather condition |
| Vehicle Type Risk | Comparative risk across Car / Motorcycle / Truck / Bus |
| Feature Importance (XGBoost) | Global model feature rankings |
| Day of Week Analysis | Weekday vs weekend accident patterns |
| Casualty Type Distribution | Breakdown of casualty categories |

### 🤖 AI Safety Chatbot
- Intent-detection chatbot covering: risk queries, SHAP explanations, heatmap guidance, emergency protocols, weather impact, safety tips, route advice
- Context-aware suggestion chips guide users through the platform

### 🚨 Emergency Alert System
- `/api/alert` simulates dispatching emergency notifications to the 3 nearest trauma centers
- Returns hospital names, phone numbers, alert status, and estimated ambulance ETA

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BROWSER (User)                       │
│  index.html · style.css · app.js                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Predict  │ │ Heatmap  │ │Analytics │ │ Chatbot  │      │
│  │  Panel   │ │  (Leaflet│ │ (Chart.js│ │  Panel   │      │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘      │
└───────┼────────────┼────────────┼─────────────┼────────────┘
        │  HTTP/REST │            │             │
┌───────▼────────────▼────────────▼─────────────▼────────────┐
│                  FastAPI Backend  (main.py)                  │
│                                                             │
│  POST /predict   ──►  XGBoost model  ──►  severity + proba  │
│  POST /explain   ──►  SHAP TreeExplainer ──► top factors    │
│  POST /chat      ──►  Intent engine  ──►  contextual reply  │
│  POST /api/alert ──►  Hospital lookup ──►  ETA + status     │
│  GET  /api/*     ──►  Pre-computed JSON analytics           │
│                                                             │
│  Assets: model.pkl · label_encoders.pkl · model_metadata.json│
│          heatmap_data.json · *_stats.json                   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                  ML Training Pipeline                        │
│                  (TrainerSet/Trainer_set_model_XGB.py)       │
│                                                             │
│  Raw CSVs  ──►  Feature Engineering  ──►  XGBoost fit       │
│            ──►  SHAP explainer setup                        │
│            ──►  Analytics JSONs export                      │
│            ──►  model.pkl + label_encoders.pkl saved         │
└─────────────────────────────────────────────────────────────┘
```

**Data flow in one sentence:** Raw accident CSVs are processed and used to train an XGBoost classifier; the trained artefacts are loaded by the FastAPI server at startup; the browser frontend calls the REST API to get predictions, explanations, and analytics data which are rendered with Chart.js and Leaflet.

---

## 🧠 How It Works — Business Logic

### 1. Severity Classification

Accident severity is mapped to three ordinal classes:

| Class | Label | Meaning |
|---|---|---|
| 0 | Minor | Low-impact incident, likely no serious injury |
| 1 | Moderate | Significant impact, possible injury |
| 2 | Severe | High-impact, likely serious injury or fatality |

### 2. Input Features

The model uses **10 features** per prediction request:

| Feature | Type | Description |
|---|---|---|
| `hour` | Numeric | Hour of day (0–23); derived from `hrmn` |
| `driver_age` | Numeric | Age of the driver |
| `engine_size` | Numeric | Vehicle engine displacement in cc |
| `car_age` | Numeric | Age of the vehicle in years |
| `weather` | Categorical | Clear / Rain / Fog / Snow / Storm |
| `lum` | Categorical | Daylight / Twilight / Night |
| `vehicle_type` | Categorical | Car / Motorcycle / Truck / Bus |
| `driver_sex` | Categorical | M / F |
| `week_day` | Categorical | Monday → Sunday |
| `state` | Categorical | Indian state name |

Categorical columns are label-encoded at training time; the same `LabelEncoder` objects are saved in `label_encoders.pkl` and reused at inference.

### 3. Prediction Flow

```
User Input (JSON)
      │
      ▼
Decode categoricals via label_encoders.pkl
      │
      ▼
Build 10-column DataFrame
      │
      ▼
model.predict()  ──►  class index (0 / 1 / 2)
model.predict_proba()  ──►  [p_minor, p_moderate, p_severe]
      │
      ▼
Return { risk_level, risk_label, probabilities, confidence }
```

### 4. SHAP Explanation Flow

```
Same 10-column DataFrame
      │
      ▼
shap.TreeExplainer(model).shap_values(df)
      │  returns per-class SHAP array
      ▼
Select SHAP array for the predicted class
      │
      ▼
Sort features by |SHAP value| descending
      │
      ▼
Return top-5 { feature, impact, direction }
```

A **positive SHAP impact** means the feature *pushes towards* the predicted class (increases risk for that class); **negative** means it *pulls away* (decreases risk).

### 5. Emergency Alert Logic

When the user clicks **Send Alert**:
1. Frontend posts current location + predicted severity to `POST /api/alert`
2. Backend selects the first 3 trauma centers from its list
3. Each receives a simulated notification with a random ETA (5–20 min)
4. Response includes ambulance dispatch status and estimated response time

---

## 🤖 ML Model & Training Pipeline

### Model: XGBoost Classifier

| Parameter | Value |
|---|---|
| Algorithm | `XGBClassifier` |
| `n_estimators` | 200 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `objective` | `multi:softmax` |
| `num_class` | 3 |
| `eval_metric` | `mlogloss` |
| Train/Test split | 80 / 20, stratified |

### Feature Importance (from trained model)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Week day | 0.1104 |
| 2 | Weather | 0.1063 |
| 3 | State | 0.1046 |
| 4 | Car age | 0.1030 |
| 5 | Vehicle type | 0.0988 |
| 6 | Driver sex | 0.0988 |
| 7 | Driver age | 0.0974 |
| 8 | Engine size | 0.0953 |
| 9 | Hour | 0.0948 |
| 10 | Light condition | 0.0905 |

### Training Steps (`TrainerSet/Trainer_set_model_XGB.py`)

1. **Load** geospatial dataset for heatmap (`AccidentsBig1.csv`) and feature dataset (`combined_accident_data.csv`)
2. **Heatmap export** — sample 2,000 geo-points, save as `backend/heatmap_data.json`
3. **Feature engineering** — extract `hour` from `hrmn`; label-encode 6 categorical columns
4. **Severity mapping** — `Minor→0, Moderate→1, Severe→2`
5. **Train/test split** — 80/20 stratified
6. **XGBoost fit** — 200 estimators, depth 6
7. **Analytics export** — hourly, weather, vehicle, day, state, casualty stats → `backend/*.json`
8. **Save artefacts** — `model.pkl`, `label_encoders.pkl`, `model_metadata.json`

An alternative `PyCaret`-based trainer (`TrainerSet/Trainmodel.py`) uses AutoML to compare and tune multiple models, useful for benchmarking.

---

## 📡 API Reference

All endpoints are served by the FastAPI backend at `http://localhost:8000`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend `index.html` |
| `POST` | `/predict` | Predict accident severity |
| `POST` | `/explain` | Get SHAP explanation for a prediction |
| `POST` | `/chat` | AI chatbot query |
| `GET` | `/api/heatmap` | Accident geo-points for heatmap |
| `GET` | `/api/feature-importance` | Model feature importance scores |
| `GET` | `/api/hourly-stats` | Average severity by hour |
| `GET` | `/api/weather-stats` | Average severity by weather |
| `GET` | `/api/vehicle-stats` | Average severity by vehicle type |
| `GET` | `/api/day-stats` | Average severity by day of week |
| `GET` | `/api/state-stats` | Average severity by Indian state |
| `GET` | `/api/casualty-stats` | Casualty type distribution |
| `GET` | `/api/model-info` | Model accuracy and training info |
| `GET` | `/api/trauma-centers` | List of nearby trauma centers |
| `POST` | `/api/alert` | Dispatch emergency alert |

### Example: `POST /predict`

**Request**
```json
{
  "hour": 22,
  "driver_age": 25,
  "engine_size": 1200,
  "car_age": 3,
  "weather": "Fog",
  "lum": "Night",
  "vehicle_type": "Motorcycle",
  "driver_sex": "M",
  "week_day": "Friday",
  "state": "Delhi"
}
```

**Response**
```json
{
  "risk_level": 2,
  "risk_label": "Severe",
  "probabilities": {
    "Minor": 0.08,
    "Moderate": 0.21,
    "Severe": 0.71
  },
  "confidence": 71.0
}
```

### Example: `POST /explain`

Returns top-5 SHAP factors for the same payload:

```json
{
  "top_factors": [
    { "feature": "weather", "impact": 0.42, "direction": "increases" },
    { "feature": "lum",     "impact": 0.31, "direction": "increases" },
    { "feature": "hour",    "impact": 0.18, "direction": "increases" },
    { "feature": "vehicle_type", "impact": 0.12, "direction": "increases" },
    { "feature": "driver_age",   "impact": -0.07, "direction": "decreases" }
  ],
  "predicted_class": 2
}
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **ML Model** | XGBoost | Gradient-boosted severity classifier |
| **Explainability** | SHAP (`TreeExplainer`) | Per-prediction feature attribution |
| **AutoML (alt)** | PyCaret | Model comparison & tuning (TrainerSet) |
| **Clustering** | scikit-learn DBSCAN | Geospatial accident cluster detection |
| **Backend** | FastAPI + Uvicorn | REST API server |
| **Data** | pandas, NumPy | Data processing and feature engineering |
| **Serialisation** | joblib | Model and encoder persistence |
| **Frontend** | HTML5 / CSS3 / Vanilla JS | Responsive single-page UI |
| **Maps** | Leaflet.js + leaflet-heat | Interactive heatmap rendering |
| **Charts** | Chart.js | Six analytics visualisation charts |
| **Fonts** | Google Fonts (Inter) | Typography |

---

## 📁 Repository Structure

```
RoadZen/
│
├── backend/                        # FastAPI server & ML artefacts
│   ├── main.py                     # Main application — all API routes
│   ├── model.pkl                   # Trained XGBoost model (binary)
│   ├── label_encoders.pkl          # Sklearn LabelEncoders for categoricals
│   ├── model_metadata.json         # Features, accuracy, feature importance
│   ├── heatmap_data.json           # Geo-points for accident heatmap
│   ├── feature_importance.json     # XGBoost feature ranking
│   ├── hourly_stats.json           # Avg severity by hour
│   ├── weather_stats.json          # Avg severity by weather condition
│   ├── vehicle_stats.json          # Avg severity by vehicle type
│   ├── day_stats.json              # Avg severity by day of week
│   ├── state_stats.json            # Avg severity by Indian state
│   └── casualty_stats.json         # Casualty type distribution
│
├── frontend/                       # Single-page web application
│   ├── index.html                  # Main HTML — layout & sections
│   ├── style.css                   # Styling, themes, responsive layout
│   └── app.js                      # API calls, chart rendering, map logic
│
├── TrainerSet/                     # Model training scripts
│   ├── Trainer_set_model_XGB.py    # ✅ Primary XGBoost training pipeline
│   ├── Trainmodel.py               # Alternative PyCaret AutoML trainer
│   ├── main.py                     # Standalone FastAPI server (earlier version)
│   └── important.md                # Quick setup notes
│
├── dataset/                        # Raw accident data
│   ├── Road-Accidents-2018-*.csv   # Ministry of Road Transport 2018 data
│   ├── Road_Accident_Profile_*.csv # City-level profiles 2011–2015
│   ├── combined_accident_data.csv  # Merged feature dataset (primary)
│   ├── combined_accident_data1.csv # Additional merged records
│   ├── only_road_accidents_data*.csv  # Filtered road-only records
│   └── labels/                     # Label reference files
│
├── .gitignore
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.9+**
- `pip` package manager
- A terminal / command prompt

### 1. Clone the Repository

```bash
git clone https://github.com/hariharasudhan-29507/RoadZen.git
cd RoadZen
```

### 2. Install Python Dependencies

```bash
pip install fastapi uvicorn xgboost shap scikit-learn pandas numpy joblib
```

> **Optional** — for the PyCaret-based alternative trainer:
> ```bash
> pip install pycaret
> ```

### 3. (Optional) Retrain the Model

Skip this step if you want to use the pre-trained `model.pkl` already in `backend/`.

Place your accident CSV files in the `dataset/` folder, then run:

```bash
cd TrainerSet
python Trainer_set_model_XGB.py
```

This will overwrite `backend/model.pkl`, `backend/label_encoders.pkl`, and all `backend/*.json` analytics files.

### 4. Start the Backend Server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be live at **`http://localhost:8000`**.

### 5. Open the Application

Navigate to **`http://localhost:8000`** in your browser. The FastAPI server serves the frontend directly — no separate web server required.

> **Interactive API docs** (Swagger UI) are available at `http://localhost:8000/docs`

### Quick Start Summary

```bash
git clone https://github.com/hariharasudhan-29507/RoadZen.git
cd RoadZen
pip install fastapi uvicorn xgboost shap scikit-learn pandas numpy joblib
cd backend
uvicorn main:app --reload
# Open http://localhost:8000
```

---

## 📊 Dataset

| File | Source | Records | Notes |
|---|---|---|---|
| `Road-Accidents-2018-Annexure-16.csv` | Ministry of Road Transport & Highways (MoRTH) | — | Annexure-level 2018 stats |
| `Road-Accidents-2018-Table-3.4.csv` | MoRTH 2018 | — | Table 3.4 breakdown |
| `Road_Accident_Profile_of_Select_Cities-2011-15.csv` | MoRTH City Profiles | — | 2011–2015 city-level |
| `combined_accident_data.csv` | Combined & cleaned | ~800–1000 rows | Primary ML training set |
| `only_road_accidents_data3.csv` | Filtered | — | Road-specific records |

The combined dataset contains features including `hrmn` (hour-minute), `driver_age`, `engine_size`, `car_age`, `weather`, `lum` (luminosity), `vehicle_type`, `driver_sex`, `week_day`, `state`, `severity`, and `casualty_type`.

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <strong>Hariharasudhan</strong><br>
      <a href="https://github.com/hariharasudhan-29507">@hariharasudhan-29507</a><br>
      <em>AI / ML Engineer · Full-Stack Developer</em>
    </td>
  </tr>
</table>

Built with ❤️ using XGBoost · SHAP · FastAPI · Leaflet.js · Chart.js

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**🛡️ RoadZen** — *Making every journey safer through AI*

📞 Emergency: **112** &nbsp;|&nbsp; 🚑 Ambulance: **108** &nbsp;|&nbsp; 🚔 Police: **100**

</div>
