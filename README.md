# 🧠 NeuroScan AI | Brain Tumor Prediction Dashboard

### Machine Learning • Python • Streamlit • Scikit-learn

<p align="center">
  An interactive Machine Learning web application that predicts brain tumor categories using patient and tumor-related data.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green" />
  <img src="https://img.shields.io/badge/Status-Development-orange" />
</p>

---

## 📌 Overview

**NeuroScan AI** is a machine learning-based healthcare analytics project developed using Python, Streamlit, and Scikit-learn.

The application allows users to enter patient information, tumor characteristics, symptoms, MRI results, and medical history. A trained Random Forest Classification model processes the input and predicts the corresponding tumor category.

The project demonstrates practical implementation of machine learning concepts, data preprocessing, classification pipelines, and interactive dashboard development.

## 🎯 Project Objectives

- Apply Machine Learning to a healthcare-related classification problem.
- Build an interactive web application using Streamlit.
- Perform data preprocessing and feature transformation.
- Implement a Random Forest classification model.
- Generate predictions from user-provided input.
- Present prediction results through an intuitive dashboard.

## ✨ Key Features

| Feature | Description |
|---|---|
| Patient Assessment | Collects patient age and gender |
| Tumor Analysis | Captures tumor size, location, histology and stage |
| Symptom Analysis | Accepts patient symptoms and MRI results |
| Medical History | Includes family history and treatment information |
| Data Preprocessing | Handles missing numerical and categorical values |
| Feature Encoding | Applies One-Hot Encoding |
| ML Prediction | Uses Random Forest Classification |
| Confidence Score | Displays the highest predicted class probability |
| Interactive UI | Responsive dashboard built with Streamlit |

## 🛠️ Tech Stack

**Programming Language**
- Python

**Libraries & Frameworks**
- Pandas – Data manipulation and analysis
- NumPy – Numerical computations
- Scikit-learn – Machine Learning and preprocessing
- Streamlit – Interactive web application

**Machine Learning Algorithms**
- Random Forest Classifier
- One-Hot Encoding
- Simple Imputation
- Scikit-learn Pipeline
- ColumnTransformer

**Development Tools**
- Jupyter Notebook
- Git
- GitHub
- VS Code

## ⚙️ Machine Learning Workflow

The project follows a structured machine learning workflow:

1. **Data Collection:** Load the brain tumor dataset using Pandas.
2. **Data Cleaning:** Remove records with missing target values.
3. **Feature Selection:** Separate independent features and the target variable.
4. **Missing Value Treatment:**
   - Numerical features: Median imputation.
   - Categorical features: Most frequent value imputation.
5. **Feature Transformation:** Apply One-Hot Encoding to categorical variables.
6. **Model Development:** Implement Random Forest Classifier with 250 estimators.
7. **Model Training:** Train the classification pipeline using the available dataset.
8. **Prediction:** Generate tumor category predictions from patient inputs.
9. **Visualization:** Display prediction results and model confidence in Streamlit.

## 📂 Project Structure

```text
streamlit-learningz/
│
├── app.py
│   └── Main Streamlit application
│
├── brain_tumor_dataset.csv
│   └── Dataset for model training
│
├── sql_project.ipynb
│   └── Jupyter Notebook
│
├── .gitignore
│
└── README.md
```

## 🚀 Installation & Setup

Follow these instructions to run the project locally.

**Step 1: Clone the Repository**

```bash
git clone https://github.com/Azmat2104/streamlit-learningz.git
```

**Step 2: Navigate to the Project**

```bash
cd streamlit-learningz
```

**Step 3: Create a Virtual Environment**

```bash
python -m venv venv
```

Activate the environment:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

**Step 4: Install Dependencies**

```bash
pip install streamlit pandas numpy scikit-learn
```

**Step 5: Run the Application**

```bash
streamlit run app.py
```

**Step 6: Open in Browser**

```text
http://localhost:8501
```

## 📊 Dataset Information

The project uses `brain_tumor_dataset.csv` for model training.

The target variable is:

```python
Tumor_Type
```

The application processes available patient and tumor-related features, including:

- Age
- Gender
- Tumor Location
- Tumor Size
- Histology
- Tumor Stage
- Symptoms
- MRI Results
- Treatment Information
- Medical History

## 💡 Skills Demonstrated

Through this project, the following technical skills are demonstrated:

- Python Programming
- Data Cleaning and Preprocessing
- Exploratory Data Handling
- Machine Learning Classification
- Feature Engineering
- Model Pipeline Development
- Predictive Analytics
- Interactive Dashboard Development
- Git and GitHub

## 🔮 Future Enhancements

- Implement train-test split and cross-validation.
- Evaluate model performance using Accuracy, Precision, Recall and F1-score.
- Add confusion matrix and classification reports.
- Improve model performance through hyperparameter tuning.
- Add feature importance and SHAP-based explainability.
- Integrate MRI image classification using Deep Learning.
- Deploy the application on Streamlit Community Cloud.

## ⚠️ Disclaimer

This project is developed strictly for educational and demonstration purposes.

It is not a clinically validated diagnostic system. The displayed model confidence is not equivalent to medical diagnostic accuracy. Predictions should not be used for medical decisions.

Always consult qualified healthcare professionals for diagnosis and treatment.

## 👨‍💻 Author

**Azmat**

GitHub: [@Azmat2104](https://github.com/Azmat2104)

Project Repository: [NeuroScan AI](https://github.com/Azmat2104/streamlit-learningz)

---

<p align="center">
  ⭐ If you find this project useful, consider giving it a star!
</p>

<p align="center">
  Built with Python, Machine Learning & ❤️
</p>
