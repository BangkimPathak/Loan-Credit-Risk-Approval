# Loan Approval Prediction using Decision Trees

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![VS Code Jupyter](https://img.shields.io/badge/VS%20Code-Jupyter%20Notebook-blueviolet.svg)](https://code.visualstudio.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning experiment and Jupyter Notebook (`loan_approval_decision_tree.ipynb`) implementing a **Decision Tree Classifier** (`DecisionTreeClassifier`) to predict **Loan Approval Status** (`0 = Rejected`, `1 = Approved`) using applicant financial and demographic data from `loan_data_new.csv`.

---

## 📌 Project Overview & Aim

Automating loan evaluation allows financial institutions to reduce credit default risks while maintaining fast, objective decision-making. This project focuses on:

1. **Dataset Anomaly Resolution**: Identifying and removing explicit data entry errors and correcting schema typos.
2. **Stratified Class Balance**: Handling target class imbalance (~77.8% Rejected vs ~22.2% Approved) via stratified sampling.
3. **Interpretable ML**: Building a Gini decision tree (`max_depth=4`) to extract transparent, human-understandable loan approval decision rules.
4. **Performance Metrics**: Evaluating model accuracy, precision, recall, F1-score, and inline confusion matrices.

---

## 📊 Dataset & Preprocessing
**Dataset Extraction 
LINK :https://www.kaggle.com/datasets/muhammadmusharraf444/loan-approval-dataset

The dataset `loan_data_new.csv` contains **45,002 initial records** with applicant financial attributes, demographics, and credit history.

### Explicit Anomaly Corrections Handled

| Anomaly Type | Original State | Action Taken | Result |
| :--- | :--- | :--- | :--- |
| **Column Name Typo** | `Home Onwership` | Renamed to `Home Ownership` | Schema corrected |
| **Age Outliers** | `Age > 100` (up to 144) | Filtered `Age <= 100` | Removed invalid age records |
| **Experience Outliers** | `Employee Experience > 60` (up to 125) | Filtered `Employee Experience <= 60` | Removed invalid experience records |
| **Categorical Features** | String labels (`Gender`, `Education`, etc.) | Applied `pd.get_dummies(drop_first=True)` | Numeric feature matrix created |
| **Class Imbalance** | ~77.8% (0) vs ~22.2% (1) | Applied `stratify=y` in `train_test_split` | Equal class ratio across splits |

---

## 📁 Repository Structure

```text
├── loan_approval_decision_tree.ipynb  # Primary VS Code Jupyter Notebook
├── loan_data_new.csv                  # Raw dataset (included in workspace)
├── requirements.txt                   # Dependency specifications
└── README.md                          # Project documentation
```
---

## 📓 Notebook Structure (`loan_approval_decision_tree.ipynb`)

The notebook follows a clear 7-step narrative structure:

- **`# Step 1: Project Overview & Aim`** — Executive summary, business context, and project goals.
- **`# Step 2: Data Loading & Initial Exploration`** — Loading data, examining `.head()`, `.info()`, `.describe()`, and target class ratios.
- **`# Step 3: Data Cleaning & Preprocessing`** — Fixing typos, filtering outliers (`Age <= 100`, `Employee Experience <= 60`), and one-hot encoding categorical variables.
- **`# Step 4: Train-Test Split & Model Training`** — Performing 80/20 stratified split (`stratify=y`, `random_state=42`) and fitting `DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)`.
- **`# Step 5: Model Evaluation & Performance Metrics`** — Computing Accuracy, Precision, Recall, F1-Score, and displaying a Seaborn Heatmap of the Confusion Matrix.
- **`# Step 6: Decision Tree Visualization`** — Plotting high-resolution decision tree diagrams with node labels and feature names via `plot_tree()`.
- **`# Step 7: Conclusion & Key Findings`** — Summarizing core decision split rules and recommendations for MLOps deployment.
---

## 🔑 Key Findings & Decision Split Rules

1. **`Loan percentage` (Loan-to-Income Ratio)**: Primary root node decision boundary. Applications exceeding specific loan-to-income thresholds are flagged for potential default.
2. **`Person Income` & `Credit Score`**: Key secondary features determining creditworthiness for approved applicants.
3. **`Home Ownership`**: Categorical status (`RENT` vs `MORTGAGE`/`OWN`) serves as a critical tertiary predictor of financial stability.

---
