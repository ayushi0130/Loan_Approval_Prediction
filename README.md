# Loan_Approval_Prediction

Welcome to the Loan Approval Prediction project! This repository contains code and documentation for predicting loan approvals based on applicant data.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Result](#Result)

## Introduction

The Loan Approval Prediction project aims to build a predictive model that can determine whether a loan application will be approved based on various features such as applicant income, loan amount, credit history, and more. This project utilizes machine learning techniques to analyze and predict loan statuses.

## Dataset

The dataset used in this project includes the following features:

- **Loan_ID**: Unique Loan ID
- **Gender**: Male/Female
- **Married**: Applicant married (Y/N)
- **Dependents**: Number of dependents
- **Education**: Applicant Education (Graduate/Undergraduate)
- **Self_Employed**: Self-employed (Y/N)
- **ApplicantIncome**: Applicant income
- **CoapplicantIncome**: Coapplicant income
- **LoanAmount**: Loan amount in thousands
- **Loan_Amount_Term**: Term of loan in months
- **Credit_History**: Credit history meets guidelines (1/0)
- **Property_Area**: Urban/Semi-Urban/Rural
- **Loan_Status**: Loan approved (Y/N)

## Project Structure

```
├── README.md
├── laon_approval_prediction
│   ├── loan.csv
│   └── loan_approval_prediction.py
└── loan_status_prediction
    ├── loan_eligibility.py
    ├── test.csv
    └── train.csv
```

- `data/`: Contains the dataset used for the project.
- `notebooks/`: Jupyter notebooks for data analysis and visualization.
- `README.md`: Project documentation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ayushi0130/Loan-Approval-Prediction.git
   cd Loan-Approval-Prediction
   ```

Create and activate a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

## Usage:

1.Run the code:

```
python loan_approval_prediction.py (for loan_approval_prediction)
python loan_status_prediction.py (for loan_status_prediction)
```

## Modeling:

this project uses a classification algorithm to predict loan approval. other algorithm can be implemented and compared for performance improvement. The modeling process includes:

- Data preprocessing
- Model training
- Model evaluation

## Result:

### loan_approval_prediction:

| Algorithm              | Prediction |
| ---------------------- | ---------- |
| RandomForestClassifier | 77%        |
| NaiveBayes             | 82%        |
| DecisionTreeClassifier | 70%        |
| KNeighborsClassifier   | 79%        |

### loan_status_prediction :

| Algorithm              | Prediction |
| ---------------------- | ---------- |
| DecisionTreeClassifier | 70%        |
| NaiveBayes             | 82%        |
