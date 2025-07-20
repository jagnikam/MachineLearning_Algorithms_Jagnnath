# Machine Learning Algorithms – Jagnnath Nikam

This repository contains Python scripts implementing five fundamental machine learning algorithms using real or built-in datasets. Each script demonstrates how to load data, train a model, and evaluate its performance. The goal is to showcase basic machine learning workflows using `scikit-learn`.

---

## 📁 Files and Algorithms

| Filename                | Algorithm               | Dataset                |
|-------------------------|--------------------------|-------------------------|
| `1_logistic_regression.py` | Logistic Regression     | Iris                    |
| `2_decision_tree.py`       | Decision Tree           | Titanic (Seaborn)       |
| `3_knn.py`                 | K-Nearest Neighbors     | Digits (sklearn)        |
| `4_svm.py`                 | Support Vector Machine  | Breast Cancer (sklearn) |
| `5_random_forest.py`       | Random Forest           | Wine (sklearn)          |

---

## 📦 Requirements

Install the following Python packages (use pip or pip3):

```bash

pip install -r requirements.txt

```

## 📁 Scripts Overview

### 1. `1_logistic_regression.py`
- **Algorithm**: Logistic Regression
- **Dataset**: Iris (built-in from scikit-learn)
- **Description**: This script classifies iris flowers into three species based on petal and sepal features using logistic regression.

---

### 2. `2_decision_tree.py`
- **Algorithm**: Decision Tree
- **Dataset**: Titanic (loaded using seaborn)
- **Description**: Predicts whether a passenger survived the Titanic disaster using features like age, sex, fare, and embarkation point. Basic preprocessing is applied to convert categorical data.

---

### 3. `3_knn.py`
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Dataset**: Digits (built-in from scikit-learn)
- **Description**: Uses the KNN algorithm to recognize hand-written digits. It classifies images based on the closest matching neighbors in the dataset.

---

### 4. `4_svm.py`
- **Algorithm**: Support Vector Machine (SVM)
- **Dataset**: Breast Cancer (built-in from scikit-learn)
- **Description**: This script classifies tumor data as malignant or benign using a linear SVM model. It highlights the ability of SVMs to handle high-dimensional data.

---

### 5. `5_random_forest.py`
- **Algorithm**: Random Forest
- **Dataset**: Wine (built-in from scikit-learn)
- **Description**: Predicts the class of wine based on chemical properties. Random Forest is an ensemble learning method that builds multiple decision trees and combines their results.

---

## 🧪 Output
Each script prints the accuracy score of the trained model on the test dataset. Sample output:
