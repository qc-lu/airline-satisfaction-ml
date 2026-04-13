# Airline Satisfaction Prediction (Machine Learning Pipeline)

## 📌 Overview
This project builds a complete machine learning pipeline to predict airline passenger satisfaction based on customer and flight-related features.

The goal is not only model performance, but also demonstrating a structured and reproducible ML workflow, including data preprocessing, feature engineering, model training, and evaluation.

---

## 📊 Dataset

The dataset used in this project is from Kaggle:

Airline Passenger Satisfaction Dataset  
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

Due to size limitations, the dataset is not included in this repository.

After downloading, place the files in:


data/
├── train.csv
└── test.csv


---

## ⚙️ Pipeline

### 1. Data Preprocessing
- Removed unnecessary columns (`id`, `Unnamed`)
- Handled missing values
- Encoded categorical variables
- Standardized features using Z-score normalization

### 2. Feature Engineering
- Removed low-impact features (e.g., gate location)
- Ensured consistent preprocessing between training and testing

### 3. Model Training
Trained multiple SVM models:

- Linear kernel
- RBF kernel
- Polynomial kernel

### 4. Evaluation
- Error rate (classification error)
- Confusion matrix

---

## 📈 Results

| Model   | Error Rate | Accuracy |
|--------|-----------|----------|
| Linear | ~0.1266   | ~87.3%   |
| RBF    | ~0.0448   | ~95.5%   |
| Poly   | ~0.0620   | ~93.8%   |

👉 RBF kernel achieved the best performance.

---

## 🔍 Confusion Matrix (RBF)

|               | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| Actual 0      | 14084       | 444         |
| Actual 1      | 716         | 10649       |

---

## 🧪 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Train models
python src/train.py
3. Evaluate models
python src/evaluate.py
📂 Project Structure
airline-satisfaction-ml/
├── data/        # dataset (not included)
├── src/         # training & evaluation scripts
├── models/      # saved models & preprocessing parameters
├── README.md
├── requirements.txt
💡 Key Learnings
Feature preprocessing has a significant impact on model performance
Model choice (kernel selection in SVM) greatly affects accuracy
Consistent preprocessing between training and testing is critical
Structuring ML code as a reproducible pipeline improves usability and clarity
🚀 Future Improvements
Hyperparameter tuning (Grid Search / CV)
Try tree-based models (XGBoost, LightGBM)
Build inference API (Flask / FastAPI)
Add cross-validation and experiment tracking
🧠 Notes

This project focuses on building a reproducible and structured ML workflow rather than only optimizing model performance.

Some computational trade-offs were made to ensure the code can run efficiently on a local machine.