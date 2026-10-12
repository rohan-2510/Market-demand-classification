# 🌾 Crop Market Demand Classification

A machine learning web application that predicts **crop price demand categories** (Low, Medium, High) for Indian agricultural markets using a **Random Forest Classifier** served via a **Flask** web interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Input Features](#input-features)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [How It Works](#how-it-works)

---

## Overview

This project addresses the problem of predicting **crop price demand levels** in Indian agricultural markets. Given environmental, logistical, and market conditions, the model classifies the expected price demand for a specific crop into one of three categories:

| Category | Description |
|----------|-------------|
| **Low**  | Low price demand range |
| **Medium** | Medium price demand range |
| **High** | High price demand range |

The target variable (`PriceCategory`) is derived from the `Price (₹/ton)` column using quantile-based binning (`pd.qcut` with 3 equal-frequency bins).

---

## Dataset

**File:** `dataset_indian_crop_price.csv`

| Property | Value |
|----------|-------|
| Total Records | **48,000** |
| Total Columns | **15** (original) → **13** after preprocessing |
| Missing Values | **None** |
| Coverage | 6 Indian states, 25 cities, 20 crop types |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `Date` | object | Month-year (dropped during preprocessing) |
| `State` | categorical | Indian state (6 states) |
| `City` | categorical | Indian city (25 cities) |
| `Crop Type` | categorical | Type of crop (20 types) |
| `Season` | categorical | Agricultural season (4 seasons) |
| `Temperature (°C)` | float | Ambient temperature |
| `Rainfall (mm)` | float | Rainfall amount |
| `Supply Volume (tons)` | float | Market supply volume |
| `Demand Volume (tons)` | float | Market demand volume |
| `Transportation Cost (₹/ton)` | float | Logistics cost |
| `Fertilizer Usage (kg/hectare)` | float | Fertilizer application rate |
| `Pest Infestation (0-1)` | float | Pest severity score |
| `Market Competition (0-1)` | float | Market competition index |
| `Price (₹/ton)` | float | **Target source** — crop price |

### Numerical Feature Statistics

| Feature | Min | Mean | Max |
|---------|-----|------|-----|
| Temperature (°C) | 15.0 | 27.5 | 40.0 |
| Rainfall (mm) | 0.0 | 150.2 | 300.0 |
| Supply Volume (tons) | 500.0 | 2741.6 | 5000.0 |
| Demand Volume (tons) | 400.0 | 2452.4 | 4499.8 |
| Transportation Cost (₹/ton) | 50.0 | 274.9 | 500.0 |
| Fertilizer Usage (kg/hectare) | 50.0 | 175.5 | 300.0 |
| Pest Infestation (0-1) | 0.0 | 0.50 | 1.0 |
| Market Competition (0-1) | 0.0 | 0.50 | 1.0 |
| Price (₹/ton) | 0.01 | 116.9 | 422.1 |

---

## ML Pipeline

The entire ML workflow is documented in `market_demand.ipynb`:

```
1. Data Loading & Exploration
        ↓
2. Preprocessing
   - Drop irrelevant columns (Date, Unnamed: 0)
   - Create target: PriceCategory = pd.qcut(Price, q=3, labels=[Low, Medium, High])
        ↓
3. Feature Engineering
   - One-Hot Encoding for: Crop Type, Season, City, State
   - Save feature column names → columns.txt
        ↓
4. Train/Test Split  (80% / 20%, stratified)
        ↓
5. Feature Scaling  (StandardScaler)
        ↓
6. Model Training   (RandomForestClassifier, random_state=42)
        ↓
7. Evaluation       (Accuracy, Confusion Matrix, Classification Report)
        ↓
8. Model Serialization
   - joblib.dump(model, crop_demand_model.pkl)
   - joblib.dump(scaler, scaler.pkl)
```

---

## Model Performance

**Algorithm:** `RandomForestClassifier` (scikit-learn, default hyperparameters)

| Metric | Score |
|--------|-------|
| **Accuracy** | **85.02%** |
| Macro Avg Precision | 0.85 |
| Macro Avg Recall | 0.85 |
| Macro Avg F1-Score | 0.85 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| High | 0.93 | 0.91 | 0.92 | 3200 |
| Low | 0.81 | 0.92 | 0.86 | 3200 |
| Medium | 0.81 | 0.72 | 0.76 | 3200 |

### Confusion Matrix

```
              Predicted
              High  Low  Medium
Actual High  [2905    5   290]
       Low   [   0 2943   257]
       Medium [ 204  682  2314]
```

---

## Project Structure

```
Market_Demand_Classification/
│
├── app.py                        # Flask web application
├── market_demand.ipynb           # Jupyter notebook (EDA + model training)
├── dataset_indian_crop_price.csv # Raw dataset (48,000 records)
├── crop_demand_model.pkl         # Trained RandomForest model (~147 MB)
├── scaler.pkl                    # Fitted StandardScaler
├── columns.txt                   # Feature column names (63 features after OHE)
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
└── templates/
    └── index.html                # Web UI for predictions
```

---

## Input Features

The web form collects the following inputs:

### Numerical Inputs

| Field | Description | Unit |
|-------|-------------|------|
| Temperature | Ambient temperature | °C |
| Rainfall | Rainfall amount | mm |
| Supply Volume | Market supply | tons |
| Demand Volume | Market demand | tons |
| Transport Cost | Logistics cost | ₹/ton |
| Fertilizer Cost | Fertilizer usage | kg/hectare |
| Pest Level | Pest infestation severity | 0–1 scale |

### Categorical Inputs (Dropdowns)

| Field | Options (in UI) |
|-------|-----------------|
| **Crop Type** | Rice, Wheat, Maize (+ 17 more supported by model) |
| **Season** | Kharif, Rabi (+ Post-Monsoon, Zaid supported by model) |
| **City** | Mumbai, Bangalore (+ 23 more supported by model) |
| **State** | Karnataka, Maharashtra (+ 4 more supported by model) |

> **Note:** The model supports all 20 crop types, 4 seasons, 25 cities, and 6 states from the training data. The current UI exposes a subset — extend the dropdowns in `index.html` to unlock full coverage.

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

> Alternatively, create a virtual environment first:
> ```bash
> python -m venv venv
> venv\Scripts\activate   # Windows
> pip install -r requirements.txt
> ```

---

## Running the App

```bash
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:5000
```

Fill in the form fields and click **Predict** to see the predicted demand category.

---

## How It Works

### Prediction Flow (app.py)

1. **User submits the form** with crop details via POST to `/predict`
2. **Numeric inputs** are parsed directly as floats
3. **Categorical inputs** are one-hot encoded:
   - e.g., `Crop Type = Rice` sets `Crop Type_Rice = 1`
4. **Missing features** (OHE columns not selected) are filled with `0`
5. **DataFrame** is constructed and columns are reordered to match training order (`columns.txt`)
6. **StandardScaler** transforms the input
7. **RandomForestClassifier** makes a prediction (Low, Medium, or High)
8. **Result** is rendered back to `index.html`

---

## Supported Categories

### Crop Types
Barley, Coffee, Cotton, Groundnut, Jute, Maize, Millets, Mustard, Onion, Potato, Pulses, Rice, Sesame, Soybean, Sugarcane, Sunflower, Tea, Tobacco, Tomato, Wheat

### Seasons
Kharif, Post-Monsoon, Rabi, Zaid

### States
Karnataka, Maharashtra, Punjab, Tamil Nadu, Uttar Pradesh, West Bengal

### Cities
Amritsar, Bangalore, Belgaum, Bellary, Chennai, Coimbatore, Davanagere, Durgapur, Gulbarga, Hubli, Kanpur, Kolkata, Lucknow, Ludhiana, Madurai, Mangalore, Mumbai, Mysore, Nagpur, Patiala, Pune, Shimoga, Siliguri, Udupi, Varanasi
