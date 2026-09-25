# DPYD Toxicity Prediction

A beginner-level machine learning project for predicting severe fluoropyrimidine toxicity using clinical and genetic data.

## Project Overview

Fluoropyrimidines such as 5-fluorouracil (5-FU) can cause severe toxicity in some patients.

The DPYD gene plays an important role in fluoropyrimidine metabolism, and some genetic variants have been associated with an increased risk of toxicity.

In this project, machine learning methods are used to explore whether clinical and genetic features can help predict severe fluoropyrimidine toxicity.

## Dataset

The dataset contains clinical and genetic information from patients treated with fluoropyrimidines.

The target variable is:

- `1` = Grade 3, 4 or 5 toxicity event
- `0` = No grade 3, 4 or 5 toxicity event

The project includes genetic features related to DPYD and other pharmacogenomic markers, together with selected clinical features.

## Machine Learning

The project includes:

- Data loading and preprocessing
- Feature selection and feature engineering
- Target variable creation
- Train/test split
- Logistic Regression
- Random Forest
- Gradient Boosting
- Handling class imbalance
- Cross-validation
- Model evaluation
- Classification report
- Confusion matrix
- ROC-AUC
- Precision-Recall analysis
- Statistical analysis
- SHAP-based feature interpretation

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- statsmodels
- SHAP
- openpyxl
- matplotlib
- seaborn

## Project Structure

```text
DPYD-Toxicity-Prediction/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/Mzhh7093/DPYD-Toxicity-Prediction.git
```

Move into the project folder:

```bash
cd DPYD-Toxicity-Prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Data Availability

The original dataset is not included in this repository.

The dataset is excluded from version control to avoid publicly sharing the original patient-level data.

## Current Status

This is a learning and portfolio project focused on applying machine learning methods to pharmacogenomics data.

The current version is an initial implementation and has not been externally validated.

The model outputs should not be used for clinical decision-making.

## Future Improvements

Possible future improvements include:

- Testing additional machine learning models
- More detailed feature engineering
- Larger pharmacogenomic datasets
- External validation
- Hyperparameter tuning
- More robust cross-validation
- Improved model interpretation

## Author

Z. Hakimi

Molecular Genetics | Bioinformatics & BioAI