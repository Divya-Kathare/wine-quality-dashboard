# 🍷 Wine Quality Analysis & Prediction Dashboard

An interactive machine learning dashboard built with Streamlit to analyze wine chemistry and predict wine quality based on key features.

---

## 🚀 What This Project Does

This project combines data analysis and machine learning into a single interactive dashboard where users can:

* Explore wine data through visualizations
* Understand relationships between chemical properties
* Get key insights affecting wine quality
* Predict wine quality in real time using a trained model

---

## 🎯 Key Features

* 📊 Dataset overview with summary metrics
* 📈 Univariate and bivariate analysis
* 🔗 Correlation heatmap
* 💡 Insight generation based on data
* 🤖 Machine learning prediction system
* 🎛️ Interactive input sliders

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas
* Plotly
* Scikit-learn
* Joblib

---

## 📂 Project Structure

```
wine-quality-dashboard/
│
├── app.py
├── wine_data.csv
├── train_model.py
├── requirements.txt
├── README.md
├── screenshots/
```

---

## 📊 Dataset Features

* Fixed Acidity – contributes to taste structure
* Volatile Acidity – high values reduce quality
* Citric Acid – adds freshness
* Residual Sugar – determines sweetness
* Chlorides – salt content
* Free Sulfur Dioxide – prevents spoilage
* Total Sulfur Dioxide – overall preservation level
* Density – relates to sugar and alcohol
* pH – acidity level
* Sulphates – improve stability
* Alcohol – major factor influencing quality

---

## 📸 Screenshots

### Overview

![Overview](screenshots/Overview.jpeg)

### Analysis

![Analysis](screenshots/Analysis_1.jpeg)

### Insights

![Insights](screenshots/Insights.jpeg)

### Prediction

![Prediction](screenshots/Prediction_2.jpeg)

---

## 🤖 Machine Learning Model

* Type: Regression Model
* Objective: Predict wine quality score (continuous value)
* Metric: R² Score

Wine quality is treated as a continuous variable, making regression more suitable than classification.

---

## ⚙️ Run Locally

```bash
git clone https://github.com/Divya-Kathare/wine-quality-dashboard.git
cd wine-quality-dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Train the Model

If the model file is not included:

```bash
python train_model.py
```

---

## 💡 Key Insights

* Higher alcohol content improves wine quality
* High volatile acidity negatively impacts taste
* Balanced pH is important for stability
* Sulphates help preserve wine
* Density influences overall body

---

## 🎯 Future Improvements

* Improve model accuracy
* Deploy dashboard online
* Enhance UI/UX
* Add real-time data

---

## 👩‍💻 Author

Divya Kathare
linkedin: https://www.linkedin.com/in/divya-kathare-41323a3a0/

---

⭐ If you found this useful, consider giving it a star!
