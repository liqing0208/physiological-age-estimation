# Physiological Age Estimation from Cardiovascular Signals (HRV + Blood Pressure)

## Overview

Chronological age does not fully reflect biological aging. This project develops a machine learning framework to estimate **physiological (biological) age** using cardiovascular features derived from **heart rate variability (HRV)** and **blood pressure (BP)**.

The project focuses on:

* Extracting aging-related signals from physiological data
* Benchmarking multiple machine learning models
* Understanding model behavior through interpretability
* Evaluating robustness under clinically relevant scenarios


## Key Contributions

* Integration of **HRV + BP features** into a unified modeling pipeline
* Engineering of **interaction features** to capture nonlinear physiological relationships
* **Systematic comparison of five machine learning models**
* **SHAP-based interpretability analysis**
* **UMAP visualization** of learned feature space
* Subgroup and clinical evaluation:

  * Sex-based error analysis
  * Age-group performance
  * Missing-age clinical estimation scenario


## Models Evaluated

The project benchmarks five models to compare linear, nonlinear, and ensemble approaches:

* **Linear Regression**

  * Baseline model for interpretability

* **Ridge Regression**

  * Regularized linear model to reduce overfitting

* **Random Forest**

  * Ensemble of decision trees capturing nonlinear relationships

* **XGBoost**

  * Gradient boosting model optimized for tabular data

* **MLP (Multi-Layer Perceptron)**

  * Neural network capturing complex feature interactions

This design enables a structured comparison across:

* Linear vs nonlinear models
* Parametric vs ensemble approaches
* Interpretable vs high-capacity models


## Methods

### 1. Feature Engineering

* **HRV features**

  * Time-domain (e.g., RMSSD, SDNN)
  * Frequency-domain (LF, HF)

* **Blood Pressure features**

  * Systolic/diastolic measures

* **Interaction features**

  * Cross-terms between HRV and BP
  * Capture physiological coupling effects


### 2. Evaluation Strategy

* Train/test split with external validation emphasis
* Metrics:

  * Prediction error (Predicted − True Age)
  * Distributional analysis

#### Subgroup Analysis

* Sex (M / F)
* Age groups (Young / Middle / Older)

#### Clinical Scenario

* Simulated **missing chronological age**
* Model predicts age purely from physiological signals


### 3. Interpretability (SHAP)

* Global feature importance ranking
* Identification of dominant physiological drivers
* Comparison of feature contributions across models (primarily XGBoost / tree-based models)


### 4. Feature Space Visualization (UMAP)

UMAP is used to explore the structure of the learned feature space:

* True age group distribution
* K-means clustering (k=3)
* Continuous age gradient

## Results

* Cardiovascular features encode meaningful **biological aging signals**
* Interaction features improve predictive performance
* Tree-based models (Random Forest, XGBoost) show strong performance on tabular data
* MLP captures nonlinear interactions but is more sensitive to data scaling
* SHAP reveals physiologically interpretable feature contributions
* UMAP shows **structured latent organization**, not random noise
* Models retain predictive ability in a **clinical missing-age scenario**


## Tech Stack

* Python
* NumPy / Pandas
* scikit-learn
* XGBoost
* SHAP
* UMAP-learn
* Matplotlib / Seaborn


## Future Work

* External dataset validation (generalization across cohorts)
* Integration with **raw ECG waveform models (CNN / Transformer)**
* Multimodal extension (clinical + imaging + genomics)
* Longitudinal modeling of aging trajectories
* Deployment as a **digital biomarker tool**



