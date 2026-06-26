# 🍷 Wine Quality Dashboard | Data Analytics & Machine Learning

> **An end-to-end Data Science project that analyzes wine quality using exploratory data analysis, machine learning, and an interactive Streamlit dashboard to uncover quality drivers and support data-driven decision-making.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn)

---

## 📖 Project Overview

Wine quality plays a critical role in customer satisfaction, pricing strategy, brand reputation, and market competitiveness. Understanding the chemical properties that influence wine quality enables wineries to optimize production and maintain consistent product standards.

This project presents an **interactive Streamlit dashboard** that combines **Exploratory Data Analysis (EDA)**, **Machine Learning**, and **interactive visualizations** to identify the factors influencing wine quality and generate actionable business insights.

The application enables users to explore the dataset, analyze quality trends, compare red and white wines, and evaluate predictive machine learning models through an intuitive web interface.

---

## 🎯 Project Objectives

* Analyze wine quality using physicochemical attributes
* Perform comprehensive exploratory data analysis
* Build machine learning models for quality prediction
* Develop an interactive dashboard for data exploration
* Generate actionable insights to support business decisions

---

## ✨ Key Features

### Interactive Dashboard

* Dynamic visualizations
* Interactive filters
* Data exploration interface
* Comparative analysis

### Exploratory Data Analysis

* Feature distribution analysis
* Correlation analysis
* Outlier detection
* Quality distribution
* Red vs White wine comparison

### Machine Learning

* Multiple classification models
* Model performance comparison
* Accuracy evaluation
* Feature importance analysis

### Business Insights

* Identify factors affecting wine quality
* Analyze chemical characteristics
* Generate production recommendations
* Support quality improvement decisions

---

## 🏗️ Project Workflow

```text
Wine Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Machine Learning Models
      │
      ▼
Model Evaluation
      │
      ▼
Interactive Streamlit Dashboard
```

---

## 📊 Dataset Overview

| Attribute       | Details                     |
| --------------- | --------------------------- |
| Domain          | Alcoholic Beverage Industry |
| Records         | 6,498                       |
| Features        | 14                          |
| Target Variable | Wine Quality                |

### Input Features

* Fixed Acidity
* Volatile Acidity
* Citric Acid
* Residual Sugar
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Density
* pH
* Sulphates
* Alcohol
* Good (Binary)
* Wine Color

### Target

* Wine Quality Score (0–10)

---

## 📈 Dashboard Modules

* Executive Overview
* Feature Analysis
* Correlation Analysis
* Chemistry Insights
* Data Explorer
* Machine Learning Results

---

## 🤖 Machine Learning Models

| Model                        | Task           |
| ---------------------------- | -------------- |
| Logistic Regression          | Classification |
| K-Nearest Neighbors (KNN)    | Classification |
| Support Vector Machine (SVM) | Classification |
| Decision Tree                | Classification |
| Random Forest                | Classification |

### Evaluation Metric

* Accuracy Score

The best-performing model was selected based on predictive performance.

---

## 🛠️ Technology Stack

| Category         | Technologies                |
| ---------------- | --------------------------- |
| Programming      | Python                      |
| Data Analysis    | Pandas, NumPy               |
| Visualization    | Plotly, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn                |
| Deployment       | Streamlit Community Cloud   |

---

## 🚀 Live Demo

Explore the deployed dashboard:

**https://wine-quality-ml-dashboard.streamlit.app/**

---

## 📂 Project Structure

```text
wine-quality-dashboard/
│
├── sprint2_dashboard.py
├── Sprint1_EDA.ipynb
├── Sprint3_Model_Building.ipynb
├── wine_quality.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### Clone the Repository

```bash
git clone https://github.com/pranjalsalvi/wine-quality-dashboard.git

cd wine-quality-dashboard
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch the Dashboard

```bash
streamlit run sprint2_dashboard.py
```

---

## 💡 Key Business Insights

* Higher alcohol content is strongly associated with better wine quality.
* Volatile acidity negatively impacts quality ratings.
* Sulphates positively influence overall wine quality.
* Red and white wines exhibit distinct chemical characteristics.
* Machine learning can effectively assist in predicting wine quality and improving production decisions.

---

## 📌 Applications

This project demonstrates practical applications in:

* Data Analytics
* Business Intelligence
* Predictive Analytics
* Quality Control
* Manufacturing Optimization
* Interactive Dashboard Development

---

## 🚀 Future Enhancements

* Advanced predictive models (XGBoost, LightGBM)
* Hyperparameter tuning
* Model explainability using SHAP
* Time-series quality monitoring
* User authentication
* Docker deployment
* Cloud deployment on AWS or Azure
* Automated report generation

---

## 👨‍💻 About Me

**Pranjal Salvi**

Aspiring **Data Analyst & AI Engineer** passionate about Data Analytics, Machine Learning, Business Intelligence, and Generative AI.

### Connect with me

* 🔗 LinkedIn: https://www.linkedin.com/in/pranjal-salvi-380732227/
* 💻 GitHub: https://github.com/pranjalsalvi

---

## ⭐ Support

If you found this project useful or interesting, consider giving it a ⭐ on GitHub.

Your support motivates me to continue building and sharing data science and AI projects.

---

### Thank you for visiting this repository! 🚀
