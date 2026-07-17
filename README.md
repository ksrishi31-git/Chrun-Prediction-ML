# Customer Churn Prediction — ML Model Comparison

Predicting customer churn using three different classifiers — **Logistic Regression**, **Decision Tree**, and **Random Forest** — built with `pandas` and `scikit-learn`.

## What it does

Each script follows the same pipeline:

1. Loads a customer churn dataset from CSV
2. Cleans the data (drops missing values)
3. Encodes categorical columns with `LabelEncoder`
4. Splits data into training and test sets (80/20)
5. Trains a classifier (Logistic Regression / Decision Tree / Random Forest)
6. Evaluates the model (accuracy, confusion matrix, classification report)
7. Predicts churn for a new/sample customer

| Script | Model | Notes |
|---|---|---|
| `logistic_regression.py` | `LogisticRegression` | Uses `StandardScaler` to scale features before training |
| `decision_tree.py` | `DecisionTreeClassifier` | `criterion='gini'`, `max_depth=5` |
| `random_forest.py` | `RandomForestClassifier` | `n_estimators=100`, `criterion='gini'`, `max_depth=5` |

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

Or, using the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Dataset

Each script expects a CSV file named something like
`customer_churn_dataset-testing-master.csv` with a `Churn` column as the target.

By default, the scripts point to a local Windows path:

```python
df = pd.read_csv(r"C:\Users\user\Downloads\customer_churn_dataset-testing-master.csv")
```

**Before running**, update this path to wherever your CSV lives — for example, put the
CSV in a `data/` folder in this repo and change the line to:

```python
df = pd.read_csv("data/customer_churn_dataset-testing-master.csv")
```

> Note: Dataset files aren't included in this repo (see `.gitignore`). You'll need to
> add your own CSV, or download the dataset separately.

## Usage

Run any script from the project folder:

```bash
python logistic_regression.py
python decision_tree.py
python random_forest.py
```

Each will print:
- The first 5 rows of the dataset
- Model accuracy
- A confusion matrix
- A full classification report
- A churn prediction for a sample new customer

## Project structure

```
customer-churn-prediction-ml/
├── logistic_regression.py   # Logistic Regression model
├── decision_tree.py          # Decision Tree model
├── random_forest.py          # Random Forest model
├── README.md                  # This file
├── requirements.txt            # Python dependencies
└── .gitignore                  # Files/folders excluded from git
```

## Notes / possible improvements

- Replace the hardcoded Windows file path with a relative path or a command-line argument
- Save trained models with `joblib`/`pickle` so they don't need retraining every run
- Add cross-validation instead of a single train/test split
- Compare model performance side-by-side in a single script or notebook
- Add feature scaling to the tree-based models for consistency (not required, but useful for comparison)
