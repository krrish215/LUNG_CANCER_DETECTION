# 🫁 Lung Cancer Detection using Machine Learning

This project uses **machine learning** to predict the likelihood of **lung cancer** based on medical and lifestyle attributes from a survey dataset.  
It was implemented and tested in **Google Colab** using Python and scikit-learn.

---

## 📘 Project Overview

The goal of this project is to develop a **predictive model** that can classify whether a person is likely to have lung cancer based on certain features such as:
- Smoking habits  
- Age  
- Anxiety  
- Chronic diseases  
- Alcohol consumption  
- Shortness of breath, etc.

---

## 📂 Dataset

**Dataset Name:** `survey lung cancer.csv`  
**Total Entries:** 309  
**Columns:** 16 (14 numerical, 2 categorical)

### Sample Columns

| Column | Type | Description |
|--------|------|-------------|
| GENDER | object | Male/Female |
| AGE | int | Age of the individual |
| SMOKING | int | Smoking habit (1–2) |
| ANXIETY | int | Level of anxiety (1–2) |
| CHRONIC DISEASE | int | Presence of chronic diseases |
| SHORTNESS OF BREATH | int | Symptom severity |
| LUNG_CANCER | object | Target variable (YES/NO) |

---

## 🧩 Steps and Methods

### 1. Data Loading and Exploration

``python
import pandas as pd
df = pd.read_csv('/content/survey lung cancer.csv')
df.info()
df.head()


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

label_encoder = LabelEncoder()
df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER'] = label_encoder.fit_transform(df['GENDER'])

X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

2. Data Preprocessing

Encoding categorical features using LabelEncoder

Splitting data into train and test sets

Applying StandardScaler for normalization


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

label_encoder = LabelEncoder()
df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER'] = label_encoder.fit_transform(df['GENDER'])

X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

⚙️ Model Training
Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

Results:

Metric	Score
Accuracy	0.9677
Precision	0.9833
Recall	0.9833
F1-score	0.9833

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
model.score(X_test, y_test)


Results:

Metric	Score
Accuracy (Test)	96.77%
Accuracy (Train)	85.02%

📊 Data Visualization

Count plots for categorical variables

Pairplot for feature relationships

Boxplot and Barplot for age distribution

Heatmap for feature correlations



from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


Classification Report

              precision    recall  f1-score   support
           0       0.00      0.00      0.00         2
           1       0.97      1.00      0.98        60
    accuracy                           0.97        62


🧠 Insights

Logistic Regression achieved the highest accuracy (~96.7%).

SVM performed similarly but slightly overfitted the training data.

Age, smoking habits, and anxiety were highly correlated with lung cancer presence.

🧰 Technologies Used

Python 3.12

Google Colab

pandas, numpy

scikit-learn

matplotlib, seaborn

🚀 How to Run

Open the Colab notebook.

Upload survey lung cancer.csv.

Run the notebook cells sequentially.

Observe model outputs and plots.
