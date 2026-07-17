# Dataset Analysis

## 1. Dataset Overview

### Dataset Name
Heart Disease Dataset (Cleveland Database)

### Source
UCI Machine Learning Repository

### Original Citation
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
Heart Disease Dataset.
UCI Machine Learning Repository.

### Problem Type
Supervised Machine Learning

### Task
Binary Classification

### Objective
Predict whether a patient is likely to have heart disease based on clinical parameters.

### Number of Instances
303

### Number of Features Used
13 clinical features (+ target variable)

### License
CC BY 4.0

### Why We Selected This Dataset

- Official academic dataset
- Widely used in machine learning research
- Well documented
- Publicly available
- Appropriate size for educational ML projects
- Contains clinically meaningful variables
- Suitable for Explainable AI (SHAP)

## Feature Analysis

Each feature in the dataset will be analyzed from three perspectives:

- Medical significance
- Machine Learning relevance
- Data quality considerations

This analysis will guide preprocessing, feature engineering, and model interpretation in later phases.

## Dataset Acquisition

### Source

UCI Machine Learning Repository

### File

processed.cleveland.data

### Storage Location

data/raw/

### Reason for Selection

The processed Cleveland dataset is the most widely used subset of the UCI Heart Disease database for machine learning research. It provides clinically meaningful features and is well documented, making it suitable for reproducible educational projects.

## Feature 1 — Age

### Description
Represents the age of the patient in years.

### Feature Type
Numerical (Continuous)

### Medical Significance
Age is a non-modifiable cardiovascular risk factor. As individuals age, structural and functional changes in the cardiovascular system can increase the likelihood of developing heart disease.

### Machine Learning Perspective
- No encoding required.
- Scaling may be required depending on the selected algorithm.
- No missing values were detected.
- Outliers will be evaluated during Exploratory Data Analysis (EDA).

### Dataset Statistics
- Minimum: 29
- Maximum: 77
- Mean: 54.44

## Feature 2 — Sex

### Description
Represents the biological sex of the patient.

### Feature Type
Categorical (Binary)

### Medical Significance
Biological sex is a clinically relevant factor in cardiovascular disease research and may influence disease prevalence and presentation.

### Machine Learning Perspective
- Already encoded.
- No additional encoding required.
- No scaling required.
- No missing values detected.

### Ethical Consideration
Although this feature may improve predictive performance, it should be interpreted carefully because demographic features can contribute to model bias. Feature importance analysis will be performed later using SHAP.

## Initial Data Quality Assessment

### Missing Values

The dataset contains a small number of missing values.

| Feature | Missing Values |
|----------|---------------:|
| ca | 4 |
| thal | 2 |

All other features contain complete data.

The missing values account for approximately 2% of the dataset and will be addressed during the preprocessing phase.

---

### Duplicate Records

No duplicate records were detected.

This indicates that each row represents a unique patient observation.