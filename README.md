# Cardiovascular Disease Classification

Machine learning project to predict cardiovascular disease using patient health data. The notebook covers the full ML workflow: preprocessing → exploratory data analysis → feature engineering → model training → hyperparameter tuning → final evaluation.

## Project Overview

- **Goal:** Build, compare, tune, and evaluate machine learning models for predicting cardiovascular disease (`cardio` target).
- **Dataset:** Patient records including age, blood pressure, cholesterol, and other health indicators.

## Workflow

1. **Preprocessing** — convert age from days to years, remove identifier columns, clean invalid blood-pressure records.
2. **Exploratory Data Analysis (EDA)** — distribution plots, correlation heatmap.
3. **Feature Engineering** — BMI, Pulse Pressure, Mean Arterial Pressure (MAP), Age².
4. **Baseline Models** — Logistic Regression, KNN, Decision Tree, Random Forest, and optionally XGBoost (using scikit-learn `Pipeline`s where scaling is required).
5. **Hyperparameter Tuning** — Grid Search with cross-validation on the training set only.
6. **Final Evaluation** — model comparison on an untouched test set using Accuracy, Precision, Recall, F1 Score, and ROC-AUC, plus confusion matrix, ROC curve, and feature importance.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- xgboost (optional)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add the dataset (`cardio_train.csv`) to the project folder.
4. Open and run the notebook:
   ```bash
   jupyter notebook Cardiovascular_Disease_Classification.ipynb
   ```

## Results

The best-performing model is selected based on test-set metrics (Accuracy, Precision, Recall, F1, ROC-AUC) rather than assumptions — see the notebook's conclusion section for the final comparison and chosen model.

## Author

Doha Anas — AI/ML Graduate, Benha National University
