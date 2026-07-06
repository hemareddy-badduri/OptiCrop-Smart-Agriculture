<div align="center">

# 🌾 OptiCrop: Smart Agricultural Production Optimization Engine

An end-to-end Machine Learning project that recommends the most suitable crop from soil nutrients and climate conditions using classification models, exploratory data analysis, a Jupyter notebook workflow, and a Flask web application.

**OptiCrop Crop Recommendation App · Python · Flask · scikit-learn**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Ready%20to%20Run-brightgreen)

https://github.com/Rajendra-523/OptiCrop-Smart-Agricultural-Production-Optimization-Engine

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Input Parameters](#-input-parameters)
- [Project Structure](#-project-structure)
- [Workflow End-to-End Pipeline](#-workflow-end-to-end-pipeline)
- [Prerequisites](#-prerequisites)
- [Installation and Setup](#-installation-and-setup)
- [Running the Project](#-running-the-project)
- [Using the Web App](#-using-the-web-app)
- [Jupyter Notebook](#-jupyter-notebook)
- [Training Details](#-training-details)
- [Model Performance](#-model-performance)
- [Testing](#-testing)
- [Customization](#-customization)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

**OptiCrop** is a smart agricultural crop recommendation system built with Machine Learning. It analyzes soil nutrient values and climate conditions, then predicts the best crop for cultivation.

This project:

- Uses a crop recommendation dataset with 7 input features and 22 crop classes.
- Performs Exploratory Data Analysis (EDA) with univariate, bivariate, and multivariate plots.
- Applies preprocessing including null checks, outlier analysis, Potassium log transformation, and train/test splitting.
- Trains multiple ML models: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and K-Means clustering.
- Saves the best model as `models/model.pkl`.
- Serves predictions through a Flask web application with Home, About, and Find Your Crop pages.
- Includes a real Jupyter notebook with runnable cells and embedded plot outputs.

---

## 🌱 Input Parameters

| Parameter | Description | Unit / Type | Typical Range |
|---|---|---:|---:|
| `N` | Nitrogen content in soil | kg/ha | 0-150 |
| `P` | Phosphorous content in soil | kg/ha | 0-150 |
| `K` | Potassium content in soil | kg/ha | 0-210 |
| `temperature` | Average temperature | °C | 5-50 |
| `humidity` | Relative humidity | % | 10-100 |
| `ph` | Soil pH value | pH | 3.5-10 |
| `rainfall` | Rainfall amount | mm | 10-300 |

### Crop Classes

The model predicts one of 22 crops, including:

`rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, and `coffee`.

---

## 📁 Project Structure

```text
OptiCrop - Smart Agricultural Production Optimization Engine/
├── config.py                       # Central paths, feature schema, constants
├── run.py                          # Main entry point to launch Flask app
├── requirements.txt                # Python dependencies
├── sample_input.json               # Example CLI prediction input
├── README.md                       # Project documentation
│
├── data/
│   ├── generate_synthetic_data.py  # Generates Crop_recommendation.csv
│   ├── Crop_recommendation.csv     # Dataset, 2200 rows and 22 crop classes
│   └── DOWNLOAD_INSTRUCTIONS.txt   # Real Kaggle dataset instructions
│
├── notebooks/
│   └── OptiCrop_EDA.ipynb          # Jupyter notebook with cell outputs and plots
│
├── src/
│   ├── preprocess.py               # Null checks, outliers, K transform, split
│   ├── train_models.py             # K-Means, classifiers, metrics, pickle model
│   └── predict_cli.py              # Command-line crop predictor
│
├── web/
│   ├── app.py                      # Flask routes and prediction logic
│   ├── static/
│   │   └── style.css               # Responsive UI styling
│   └── templates/
│       ├── base.html               # Base layout
│       ├── home.html               # Home page
│       ├── about.html              # About page
│       └── find_your_crop.html     # Prediction form page
│
├── models/
│   └── model.pkl                   # Saved trained model bundle
│
└── outputs/
    ├── model_metrics.csv           # Model comparison metrics
    ├── descriptive_stats.csv       # Descriptive statistics
    ├── mean_per_crop.csv           # Mean features per crop
    └── figures/                    # EDA and model evaluation plots
```

---

## 🔄 Workflow End-to-End Pipeline

```text
┌────────────────────────────────────────────────────────────────────┐
│                         OPTICROP PIPELINE                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. Dataset Preparation                                             │
│     └── data/generate_synthetic_data.py                             │
│         └── Creates Crop_recommendation.csv                         │
│             → 2200 rows × 8 columns                                 │
│             → 7 features + crop label                               │
│                                                                    │
│  2. Exploratory Data Analysis                                       │
│     └── notebooks/OptiCrop_EDA.ipynb                                │
│         ├── Dataset overview                                        │
│         ├── Null value analysis                                     │
│         ├── Univariate plots                                        │
│         ├── Bivariate plots                                         │
│         ├── Correlation heatmap                                     │
│         └── Descriptive statistics                                  │
│                                                                    │
│  3. Preprocessing                                                   │
│     └── src/preprocess.py                                           │
│         ├── Checks missing values                                   │
│         ├── Detects outliers using IQR                              │
│         ├── Applies log1p transform on Potassium (K)                │
│         ├── Groups crops by season                                  │
│         └── Splits data into train/test sets                        │
│                                                                    │
│  4. Model Training                                                  │
│     └── src/train_models.py                                         │
│         ├── StandardScaler feature scaling                          │
│         ├── K-Means clustering + Elbow Method                       │
│         ├── Logistic Regression                                     │
│         ├── K-Nearest Neighbors                                     │
│         ├── Decision Tree                                           │
│         ├── Random Forest                                           │
│         ├── Accuracy / Precision / Recall / F1 evaluation           │
│         └── Saves best model → models/model.pkl                     │
│                                                                    │
│  5. Prediction                                                       │
│     ├── src/predict_cli.py                                          │
│     │   └── Predict crop from terminal or JSON input                 │
│     └── web/app.py                                                  │
│         └── Predict crop from Flask web form                        │
│                                                                    │
│  6. Web Application                                                  │
│     └── run.py                                                      │
│         └── Launches Flask app at http://127.0.0.1:5000             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## ✅ Prerequisites

- Python 3.10 or above
- pip package manager
- Modern browser such as Chrome, Edge, Firefox, or Safari
- VS Code or Jupyter Notebook for notebook execution

---

## 📦 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Rajendra-523/OptiCrop-Smart-Agricultural-Production-Optimization-Engine
cd OptiCrop-Smart-Agricultural-Production-Optimization-Engine
```

If you are using the current local folder directly:

```powershell
cd "D:\OptiCrop - Smart Agricultural Production Optimization Engine"
```

### 2. Create a Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

For Git Bash:

```bash
python -m venv venv
source venv/Scripts/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| Flask | Web application framework |
| pandas | Dataset handling and analysis |
| numpy | Numerical computation |
| scikit-learn | Machine Learning models and metrics |
| matplotlib | Plot generation |
| seaborn | Statistical visualizations |
| ipykernel / notebook | Jupyter notebook execution |
| joblib | Model utility dependency |

---

## 🚀 Running the Project

### Option 1: Run the Full Pipeline

```powershell
# Step 1: Generate dataset
python data\generate_synthetic_data.py

# Step 2: Run EDA script and save plots
python notebooks\eda.py

# Step 3: Train ML models and save the best model
python src\train_models.py

# Step 4: Start the Flask app
python run.py
```

Open in browser:

[Open the app](http://127.0.0.1:5000)

### Option 2: Quick Start App Only

If `data/Crop_recommendation.csv` and `models/model.pkl` already exist:

```powershell
python run.py
```

Then open:

[Open the app](http://127.0.0.1:5000)

### Option 3: Use the Command-Line Predictor

```powershell
python src\predict_cli.py --json sample_input.json
```

Or pass values manually:

```powershell
python src\predict_cli.py --N 80 --P 48 --K 40 --temperature 23 --humidity 82 --ph 6.4 --rainfall 225
```

Example output:

```text
Recommended crop: RICE
```

---

## 🌐 Using the Web App

### Home Page (`/`)

The home page introduces OptiCrop and provides navigation to the crop prediction form.

### About Page (`/about`)

The about page explains the agricultural problem, project approach, input features, and impact.

### Find Your Crop Page (`/find-your-crop`)

Enter the 7 input values:

- Nitrogen (N)
- Phosphorous (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

Click **Predict** to get the recommended crop.

### Example Prediction

| N | P | K | Temperature | Humidity | pH | Rainfall | Predicted Crop |
|---:|---:|---:|---:|---:|---:|---:|---|
| 80 | 48 | 40 | 23.0 | 82.0 | 6.4 | 225.0 | Rice |

---

## 📓 Jupyter Notebook

The project includes a real Jupyter notebook:

```text
notebooks/OptiCrop_EDA.ipynb
```

It contains separate runnable cells for:

- Importing libraries
- Loading the dataset
- Dataset overview
- Null value analysis
- Crop class distribution
- Univariate plots
- Bivariate plots
- Correlation heatmap
- Pairplot
- Descriptive statistics

Run it with:

```powershell
python -m notebook notebooks\OptiCrop_EDA.ipynb
```

You can also open it directly in VS Code and run each cell one by one.

---

## 🧠 Training Details

### Dataset

| Property | Value |
|---|---|
| Dataset file | `data/Crop_recommendation.csv` |
| Rows | 2200 |
| Columns | 8 |
| Features | 7 numerical inputs |
| Target | `label` |
| Crop classes | 22 |

### Preprocessing

- Checks missing values using `df.isnull().sum()`.
- Detects outliers with the IQR method.
- Applies `np.log1p()` to Potassium (`K`) because it has high right skew.
- Splits the dataset into train/test sets with stratification.
- Standardizes features with `StandardScaler`.

### Models Used

| Model | Type | Purpose |
|---|---|---|
| K-Means | Unsupervised | Cluster similar agricultural conditions |
| Logistic Regression | Supervised classification | Best-performing classifier in current run |
| K-Nearest Neighbors | Supervised classification | Similarity-based crop prediction |
| Decision Tree | Supervised classification | Rule-based non-linear prediction |
| Random Forest | Supervised classification | Ensemble tree-based prediction |

The best model is selected using **macro-F1 score** and saved as:

```text
models/model.pkl
```

---

## 📊 Model Performance

Current model comparison from `outputs/model_metrics.csv`:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9773 | 0.9791 | 0.9773 | 0.9770 |
| K-Nearest Neighbors | 0.9682 | 0.9689 | 0.9682 | 0.9680 |
| Decision Tree | 0.9409 | 0.9456 | 0.9409 | 0.9405 |
| Random Forest | 0.9727 | 0.9734 | 0.9727 | 0.9726 |

**Best Model:** Logistic Regression  
**Best Macro-F1:** 0.9770

Generated evaluation outputs include:

- `outputs/model_metrics.csv`
- `outputs/figures/kmeans_elbow.png`
- `outputs/figures/confusion_logistic_regression.png`
- `outputs/figures/confusion_k-nearest_neighbors.png`
- `outputs/figures/confusion_decision_tree.png`
- `outputs/figures/confusion_random_forest.png`

---

## 🧪 Testing

### Run Python Compilation Check

```powershell
python -m compileall src web run.py config.py
```

### Test CLI Prediction

```powershell
python src\predict_cli.py --json sample_input.json
```

Expected result for the included sample:

```text
Recommended crop: RICE
```

### Test Flask Routes Manually

Start the app:

```powershell
python run.py
```

Open the app:

[Open the app](http://127.0.0.1:5000)

---

## 🛠 Customization

### Use the Real Kaggle Dataset

Download the real CSV using the instructions in:

```text
data/DOWNLOAD_INSTRUCTIONS.txt
```

Then replace:

```text
data/Crop_recommendation.csv
```

Retrain:

```powershell
python src\train_models.py
```

### Add a New Model

Edit:

```text
src/train_models.py
```

Add the model inside `build_classifiers()`, then rerun:

```powershell
python src\train_models.py
```

### Change Form Fields

If you add more features, update:

- `config.py`
- `src/preprocess.py`
- `src/train_models.py`
- `web/app.py`
- `web/templates/find_your_crop.html`

---

## 🛟 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Dataset not found | Run `python data\generate_synthetic_data.py` |
| Model not found | Run `python src\train_models.py` |
| Port 5000 already in use | Set another port, e.g. `$env:PORT=5001; python run.py` |
| Notebook does not open | Run `pip install notebook ipykernel` |
| Prediction error | Check all 7 form values are valid numbers |

---

## 🙏 Credits

- Dataset inspiration: public Crop Recommendation dataset from Kaggle.
- Built with Flask, scikit-learn, pandas, numpy, matplotlib, seaborn, and Jupyter Notebook.

<div align="center">

**🌾 OptiCrop - grow the right crop with data-driven intelligence.**

</div>
