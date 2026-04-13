# Airline Satisfaction Prediction (SVM)

## 📌 Overview
This project aims to predict airline passenger satisfaction using machine learning models based on customer and flight-related features.

The focus is on building a complete machine learning pipeline, including data preprocessing, feature engineering, model training, and evaluation.

---

## 📊 Dataset
- Source: Kaggle Airline Passenger Satisfaction dataset
- Data includes passenger attributes such as:
  - Customer type
  - Travel type
  - Seat class
  - Service ratings
- Target:
  - Satisfaction (1: satisfied, 0: neutral or dissatisfied)

---

## ⚙️ Method

### 1. Data Preprocessing
- Removed unnecessary columns (`id`, `Unnamed`)
- Handled missing values
- Encoded categorical variables into numerical form
- Applied Z-score normalization

### 2. Feature Engineering
- Dropped low-impact features (e.g., gate location)
- Ensured consistent preprocessing between train and test data

### 3. Model Training
Trained multiple SVM models with different kernels:

- Linear kernel
- RBF kernel
- Polynomial kernel

### 4. Evaluation
- Accuracy (error rate)
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

## 🔍 Confusion Matrix (Example: RBF)

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
```bash
python src/train.py

3. Evaluate models
```bash
python src/evaluate.py

📂 Project Structure
airline-satisfaction-ml/
├── data/
├── src/
├── models/
├── README.md
├── requirements.txt
💡 Key Learnings
Feature preprocessing significantly impacts model performance
Kernel selection plays a critical role in SVM effectiveness
Ensuring consistent preprocessing between train and test data is essential
Engineering a reproducible ML pipeline is as important as model accuracy
🚀 Future Improvements
Hyperparameter tuning (Grid Search)
Replace SVM with tree-based models (e.g., XGBoost)
Build a simple API for inference
Add cross-validation
🧠 Notes

Due to the relatively large dataset size, training time varies across kernels.
This repository focuses on building a reproducible and structured ML workflow rather than optimizing computational performance.