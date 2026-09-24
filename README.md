# DPYD Toxicity Prediction

A beginner-level machine learning project for predicting severe fluoropyrimidine toxicity using clinical and genetic data.

## Project Overview

Fluoropyrimidines such as 5-fluorouracil (5-FU) can cause severe toxicity in some patients.

The DPYD gene is involved in the metabolism of fluoropyrimidines, and some genetic variants may be associated with an increased risk of toxicity.

In this project, machine learning models are used to explore whether clinical and genetic features can help predict severe fluoropyrimidine toxicity.

## Dataset

The dataset contains clinical and genetic information from patients treated with fluoropyrimidines.

The target variable is:

- `1` = Grade 3, 4 or 5 toxicity event
- `0` = No grade 3, 4 or 5 toxicity event

The project also includes genetic features related to DPYD and other pharmacogenomic markers.

## Machine Learning

The project includes:

- Data loading and preprocessing
- Target variable creation
- Feature selection
- Train/test split
- Logistic Regression
- Handling class imbalance
- Model prediction
- Classification report
- Confusion matrix
- ROC-AUC evaluation

The main libraries used are:

- Python
- pandas
- scikit-learn
- openpyxl

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

## Current Status

This is a learning and portfolio project focused on applying machine learning concepts to pharmacogenomics data.

The current version is an initial implementation and is not intended for clinical decision-making.

## Future Improvements

Possible future improvements include:

- Testing additional machine learning models
- More detailed feature engineering
- Cross-validation
- Hyperparameter tuning
- Better evaluation of model performance
- Working with larger pharmacogenomic datasets

## Author

Z. Hakimi

Molecular Genetics | Bioinformatics & BioAI