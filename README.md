# 🍷 Wine Quality Dashboard

### Interactive Streamlit Dashboard for Wine Quality Analysis, EDA & Machine Learning

🔗 **Live Demo:** https://wine-quality-ml-dashboard.streamlit.app/

📂 **GitHub Repository:** https://github.com/pranjalsalvi/wine-quality-dashboard

---

## 📖 Overview

The **Wine Quality Dashboard** is an end-to-end Data Science project that analyzes wine quality using physicochemical properties and visualizes insights through an interactive Streamlit application.

The project combines:

* 📊 Exploratory Data Analysis (EDA)
* 📈 Interactive Data Visualization
* 🤖 Machine Learning Modeling
* 💡 Business Recommendations

to help wineries identify the key factors influencing wine quality and support data-driven decision-making.

---

## 🎯 Business Objective

Wine quality directly impacts:

* Customer Satisfaction
* Brand Reputation
* Market Competitiveness
* Product Pricing
* Revenue Growth

This project helps identify the chemical characteristics that contribute to wine quality and provides actionable insights for improving production processes.

---

## 📊 Dataset Information

| Attribute       | Details                   |
| --------------- | ------------------------- |
| Domain          | Alcoholic Beverage Sector |
| Records         | 6,498                     |
| Features        | 14                        |
| Target Variable | Wine Quality Score        |

### Input Variables

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
* Good (1/0)
* Color (Red / White)

### Target Variable

* Quality (Score between 0–10)

---

## 🔍 Exploratory Data Analysis

The EDA phase focused on understanding the distribution of wine attributes and identifying factors influencing wine quality.

### Analysis Performed

✅ Feature Distribution Analysis

✅ Correlation Analysis

✅ Quality Distribution Analysis

✅ Red vs White Wine Comparison

✅ Outlier Detection

✅ Feature Importance Analysis

✅ Business Insights & Recommendations

### Key Insights

* Higher alcohol content is strongly associated with better wine quality.
* Volatile acidity negatively impacts quality ratings.
* Sulphates contribute positively to wine quality.
* Red and white wines exhibit different chemical profiles.
* High-quality wines generally maintain better chemical balance.

---

## 📈 Streamlit Dashboard Features

### Dashboard Modules

* 🏠 Overview Dashboard
* 📊 Feature Analysis
* 🔗 Correlation Analysis
* 🧪 Chemistry Insights
* 📋 Data Explorer

### Interactive Features

* Wine Color Filter
* Quality Range Selection
* Dynamic Visualizations
* Correlation Heatmaps
* Comparative Analysis
* Download Filtered Dataset

---

## 🤖 Machine Learning Models

The following algorithms were trained and evaluated:

| Algorithm                    | Purpose        |
| ---------------------------- | -------------- |
| K-Nearest Neighbors (KNN)    | Classification |
| Logistic Regression          | Classification |
| Support Vector Machine (SVM) | Classification |
| Decision Tree                | Classification |
| Random Forest                | Classification |

### Evaluation Metric

**Accuracy Score**

The best-performing model was selected based on predictive performance.

---

## 🛠️ Tech Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn

### Deployment

* Streamlit Community Cloud

---

## 🚀 Live Dashboard

Explore the deployed application:

👉 https://wine-quality-ml-dashboard.streamlit.app/

---

## ⚙️ Installation & Setup

### Clone Repository

```bash
git clone https://github.com/pranjalsalvi/wine-quality-dashboard.git
cd wine-quality-dashboard
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run sprint2_dashboard.py
```

---

## 📁 Project Structure

```text
wine-quality-dashboard/
│
├── sprint2_dashboard.py
├── wine_quality.csv
├── requirements.txt
├── README.md
│
├── Sprint1_EDA.ipynb
├── Sprint3_Model_Building.ipynb
```

---

## 💡 Business Recommendations

* Optimize alcohol content and sulphate levels to improve wine quality.
* Closely monitor volatile acidity during production.
* Implement predictive quality monitoring systems.
* Utilize machine learning models for quality forecasting and process optimization.

---

## 👨‍💻 Author

### Pranjal Salvi

🔗 LinkedIn: https://www.linkedin.com/in/pranjal-salvi-380732227

🔗 GitHub: https://github.com/pranjalsalvi

---

### ⭐ If you found this project helpful, consider giving it a star!
