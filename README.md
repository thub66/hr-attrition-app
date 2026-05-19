# 👥 HR Employee Attrition Prediction

A Streamlit web app that predicts whether an employee is likely to leave a company using Machine Learning.

## 🚀 Live Demo
[Click here to open the app](https://your-app-name.streamlit.app) ← replace after deployment

---

## 📁 Project Structure
```
hr-attrition-app/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── HR-Employee-Attrition.csv     # Dataset (IBM HR Analytics)
├── attrition_pipeline.pkl        # Trained ML pipeline (StandardScaler + RandomForest)
└── README.md
```

---

## ⚙️ Features
- **🔍 Predict Attrition** — Enter employee details and get instant predictions with probabilities
- **📊 EDA Dashboard** — Explore attrition patterns through interactive visualizations
- **📈 Model Performance** — Compare 7 classification models side by side

---

## 🧠 Models Trained
| Model | Accuracy |
|---|---|
| Gradient Boosting | 88.1% ✅ Best |
| SVM | 87.4% |
| Random Forest | 87.1% |
| XGBoost | 87.1% |
| Decision Tree | 81.0% |
| Logistic Regression | 76.2% |
| KNN | 64.3% |

---

## 🛠️ Run Locally
```bash
git clone https://github.com/your-username/hr-attrition-app.git
cd hr-attrition-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Dataset
IBM HR Analytics Employee Attrition dataset — 1470 employees, 35 features.
