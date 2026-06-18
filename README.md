# ❤️ Cardiovascular Disease Prediction

Machine learning project that predicts cardiovascular disease risk using patient demographic, lifestyle, and clinical measurements.

The model was trained using XGBoost on a cleaned version of the Cardiovascular Disease Dataset and later integrated into a FastAPI inference service with explainability support.

**Tech Stack:** Python, Pandas, Scikit-Learn, XGBoost, AWS S3, AWS SageMaker Studio

---

## 📂 Repository Structure

```text
├── CardiacDataEDA.ipynb      # Data cleaning, EDA, scaling, train/test split
├── XGBoost_Train.ipynb       # Model training and evaluation
```


## 📊 Dataset

**Source:** Cardiovascular Disease Dataset (Kaggle)

* Original dataset: 70,000 patient records
* Target variable: `cardio`

  * `0` = No cardiovascular disease
  * `1` = Cardiovascular disease

### Features

| Feature     | Description              |
| ----------- | ------------------------ |
| age         | Age (years)              |
| gender      | Biological sex           |
| height      | Height (cm)              |
| weight      | Weight (kg)              |
| ap_hi       | Systolic blood pressure  |
| ap_lo       | Diastolic blood pressure |
| cholesterol | Cholesterol level        |
| gluc        | Glucose level            |
| smoke       | Smoking status           |
| alco        | Alcohol consumption      |
| active      | Physical activity status |

---

##  Data Cleaning Pipeline

The following preprocessing steps were applied:

1. Removed patient ID column
2. Converted age from days to years
3. Performed exploratory data analysis (EDA)
4. Removed physiologically unrealistic values:

   * Height: 100–220 cm
   * Weight: 30–200 kg
   * Systolic BP: 70–250
   * Diastolic BP: 40–150
5. Applied Min-Max scaling to:

   * age
   * height
   * weight
   * ap_hi
   * ap_lo

### Final Dataset

| Metric           | Value  |
| ---------------- | ------ |
| Original Records | 70,000 |
| Cleaned Records  | 68,676 |
| Features         | 11     |
| Target           | cardio |

---

## 🔍 Exploratory Data Analysis Findings

Key observations from the dataset:

* Age and cholesterol showed the strongest correlation with cardiovascular disease.
* Cholesterol displayed a clear risk gradient:

  * Normal: ~44% disease rate
  * Above Normal: ~60%
  * Well Above Normal: ~77%
* Target classes were nearly balanced:

  * Positive: 50.03%
  * Negative: 49.97%

No class balancing techniques were required.

---

## 📈 Train-Test Split

| Split    | Records |
| -------- | ------- |
| Training | 54,940  |
| Testing  | 13,736  |

* Train/Test Ratio: 80/20
* Random Seed: 42

Data was exported to AWS S3 for model training.

---

##  Model Training

### Algorithm

**XGBoost Classifier**

Objective:

```python
binary:logistic
```

Evaluation Metric:

```python
auc
```

### ⚙️ Hyperparameter Search

Grid search evaluated 27 configurations across:

```text
max_depth       = [3, 5, 7]
eta             = [0.05, 0.1, 0.2]
num_boost_round = [50, 100, 200]
```

### 🏆 Best Model

| Parameter       | Value |
| --------------- | ----- |
| max_depth       | 3     |
| eta             | 0.1   |
| num_boost_round | 200   |

---

## 📋 Model Performance

### Baseline Model

| Metric         | Value |
| -------------- | ----- |
| Train AUC      | 0.820 |
| Validation AUC | 0.799 |

### Final Tuned Model

| Metric               | Value     |
| -------------------- | --------- |
| Validation AUC       | **0.802** |
| Train-Validation Gap | **0.005** |

The tuned model significantly reduced overfitting while slightly improving validation performance.

---

## 🎯 Classification Performance

### Threshold = 0.4 (Deployment Threshold)

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 72.6% |
| Precision | 69.9% |
| Recall    | 78.3% |

### Threshold = 0.5

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 73.7% |
| Precision | 75.4% |
| Recall    | 69.7% |

The deployment threshold was set to **0.4** to prioritize recall and reduce missed disease cases.

---

## ☁️ Development Environment

Training was performed in AWS SageMaker Studio using:

```text
xgboost
scikit-learn
pandas
numpy
matplotlib
seaborn
boto3
```

Instance Type:

```text
ml.t3.medium
```

---

## Project Outcome

This project demonstrates an end-to-end machine learning workflow:

* Data cleaning and preprocessing
* Exploratory data analysis
* Feature scaling
* Hyperparameter tuning
* XGBoost model training
* Model evaluation using AUC
* Model deployment preparation for API inference

The trained model was later integrated into a FastAPI service that provides cardiovascular disease risk predictions and explainability support.
