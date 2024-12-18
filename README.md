# Mental Health Survey Depression Classification

## Overview

This repository contains a machine learning project that uses data from a mental health survey to predict whether individuals experience depression. The project employs various preprocessing techniques and classification models to achieve accurate predictions. This is a Kaggle playground prediction competition where submissions are evaluated using **Accuracy Score**.

## Goal

Explore factors that may cause individuals to experience depression and build a predictive model using **Gradient Boosting** and **Random Forest** classifiers.

## Dataset

### Files:
- **`train.csv`**: Training dataset with labeled data.
- **`test.csv`**: Test dataset for making predictions.
- **`submission_project.csv`**: Submission file containing predicted results.
- **`final_code_notebook.ipynb`**: Jupyter Notebook with the implementation.
- **`final_code.py`**: Python Script
- **`project_presentation_cs437.pptx`**: Presentation File


### Data Columns:
- **Features**: Various demographic and lifestyle attributes, including:
  - `Age`
  - `Sleep Duration`
  - `Dietary Habits`
  - `Profession`
  - `Degree`
  - `City`
  - **Ordinal Columns**: `Academic Pressure`, `Work Pressure`, `Study Satisfaction`, `Job Satisfaction`, `Financial Stress`
- **Target**: `Depression` (0 = No Depression, 1 = Depression)

## Dependencies

Install the required Python libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Key Steps

1. **Data Preprocessing**:
   - Handle missing values using `SimpleImputer`.
   - Map categorical values to broader categories for `Sleep Duration`, `Dietary Habits`, `Profession`, and `Degree`.
   - Standardize numerical features and one-hot encode categorical features.

2. **Feature Selection**:
   - Use `SelectKBest` with `f_classif` to select the most important features.

3. **Model Training**:
   - Train models using `GradientBoostingClassifier` and `RandomForestClassifier`.
   - Perform hyperparameter tuning with `RandomizedSearchCV`.

4. **Evaluation**:
   - Evaluate the best model using:
     - **Accuracy Score**
     - **Classification Report**
     - **Confusion Matrix**

5. **Prediction**:
   - Generate predictions for the test dataset.
   - Export the results to `submission_project.csv`.

## Results

- **Best Accuracy Score**: Displayed after hyperparameter tuning.
- **Confusion Matrix**: Visualizes model performance on the training data.

## Sample Output

```
Best Parameters: {'select__k': 10, 'classifier': GradientBoostingClassifier(), ...}
Best Accuracy Score: 0.82

Classification Report:
              precision    recall  f1-score   support
No Depression       0.85      0.88      0.86       115133
   Depression       0.78      0.73      0.75       25567
```

## Competition Performance
- **Position**: 1229th out of 2685 participants
- **Percentile**: 54th percentile

## License
This project is licensed under the MIT License.

## Acknowledgments
- Dataset Source: [Kaggle Playground Series - Exploring Mental Health Data](https://www.kaggle.com/competitions/playground-series-s4e11)
- Instructions provided by the course professor.
