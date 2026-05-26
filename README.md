# 📈 Sales Forecasting / Retail Demand Prediction Dashboard

A beginner-friendly Machine Learning dashboard that predicts **retail sales demand** using historical sales data.  
The project analyzes monthly sales trends, seasonality, actual vs predicted sales, and provides simple business insights for inventory planning.

---

## 📌 Project Overview

The **Sales Forecasting / Retail Demand Prediction Dashboard** is a Python-based Machine Learning project built to understand retail sales patterns and predict future demand.

This project uses a retail and warehouse sales dataset containing information such as:

- Year
- Month
- Supplier
- Item Code
- Item Description
- Item Type
- Retail Sales
- Retail Transfers
- Warehouse Sales

The main goal of this project is to predict **Retail Sales** using available sales-related features.

---

## 🎯 Objective

The objective of this project is to build a simple and practical sales prediction system that can:

- Analyze monthly retail sales trends
- Identify seasonal sales patterns
- Predict retail demand using Machine Learning
- Compare actual sales with predicted sales
- Generate business insights for inventory planning

---

## 📂 Dataset

The dataset used in this project is:

It is placed inside the data folder.

data/Retail_and_wherehouse_Sale.csv


| Column Name      | Description                    |
| ---------------- | ------------------------------ |
| YEAR             | Year of sales record           |
| MONTH            | Month of sales record          |
| SUPPLIER         | Supplier name                  |
| ITEM CODE        | Unique item/product code       |
| ITEM DESCRIPTION | Product description            |
| ITEM TYPE        | Type/category of item          |
| RETAIL SALES     | Retail sales value             |
| RETAIL TRANSFERS | Retail transfer quantity/value |
| WAREHOUSE SALES  | Warehouse sales value          |


🧠 Target Variable

The target variable used for prediction is:

RETAIL SALES

The model predicts retail sales demand based on other available features.

🛠️ Tech Stack
Technology	Purpose
Python	Main programming language
Pandas	Data loading and preprocessing
Matplotlib	Data visualization
Scikit-learn	Machine Learning model training
Streamlit	Web dashboard development
Joblib	Saving and loading trained model
📁 Project Structure
Retail Demand Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Retail_and_wherehouse_Sale.csv
│
└── models/
    ├── sales_model.pkl
    ├── feature_columns.pkl
    └── target_col.pkl
