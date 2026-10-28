# 🛍️ Real-Life Retail Sales Prediction Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning to predict customer purchase channels based on customer profiles, product details, pricing, discounts, and transaction information. It predicts whether a customer is more likely to purchase through a digital platform (online/mobile app) or a physical retail store.

An interactive web application is developed using Streamlit to display predictions, probability scores, and marketing recommendations.

## 🎯 Objectives

* Predict customer purchase channels using Machine Learning.
* Analyze customer and product information.
* Calculate sales, cost, and profit.
* Display prediction probabilities for online and in-store purchases.
* Support data-driven marketing decisions.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

## ✨ Key Features

* Interactive customer input form.
* Digital buyer vs. physical retail buyer prediction.
* Purchase channel probability scores.
* Sales, cost, and profit calculation.
* Marketing recommendations based on predictions.
* User-friendly interface with a light theme.

## 🤖 Machine Learning Model

**Algorithm:** XGBoost Classifier

The trained model and preprocessor are loaded into the Streamlit application to process customer inputs and generate purchase channel predictions.

## 📊 Prediction Output

The application displays:

* Predicted purchase channel.
* Digital purchase probability.
* In-store purchase probability.
* Customer and transaction input summary.
* Suggested marketing actions.

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd YOUR_PROJECT_FOLDER
```

### 3. Install Required Libraries

```bash
pip install streamlit pandas scikit-learn xgboost joblib
```

### 4. Run the Application

```bash
streamlit run main.py
```

## 📁 Project Structure

```text
Retail-Sales-Prediction/
│
├── main.py
├── real_life_retail_sales_ML.csv
├── xgboost_retail_model.joblib
├── retail_preprocessor.joblib
└── README.md
```

*Note: Update the filenames according to your actual project files.*

## 💡 Business Benefits

* Helps identify potential digital and in-store customers.
* Supports targeted marketing campaigns.
* Provides insights for promotional planning.
* Helps businesses make informed marketing decisions.

## 👩‍💻 Author

**Keerthi Reddy**

## 🏷️ Project Tags

`Machine Learning` `Python` `XGBoost` `Retail Analytics` `Sales Prediction` `Streamlit` `Data Analytics`
