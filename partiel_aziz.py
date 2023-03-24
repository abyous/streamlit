import pandas as pd
from imblearn.over_sampling import SMOTE

# Charger le jeu de données dans un dataframe
data = pd.read_csv('dataset.csv')

# Vérifier les doublons
duplicates = data.duplicated()
print("Nombre de doublons : ", duplicates.sum())

# Supprimer les doublons
data.drop_duplicates(inplace=True)

# Vérifier que les doublons ont été supprimés
duplicates = data.duplicated()
print("Nombre de doublons après suppression : ", duplicates.sum())

# Vérifier les valeurs manquantes dans chaque colonne
null_values = data.isnull().sum()
print(null_values)

# Remplacer les valeurs manquantes de la colonne "Course" par la moyenne de la colonne
data['Course'].fillna(data['Course'].mean(), inplace=True)

# Supprimer les lignes ayant des valeurs manquantes dans les autres colonnes
data.dropna(inplace=True)

# Vérifier que toutes les valeurs manquantes ont été traitées
null_values = data.isnull().sum()
print(null_values)

# Séparer les caractéristiques et les étiquettes
X = data.drop(['Target', 'Debtor'], axis=1)
y = data['Target']

# Appliquer SMOTE pour équilibrer les classes
smote = SMOTE()
X, y = smote.fit_resample(X, y)

# Compter le nombre d'observations dans chaque classe
print(y.value_counts())

# Encoder les variables catégorielles en utilisant l'encodage one-hot
encoded_data = pd.get_dummies(X, columns=['Scholarship holder', 'Displaced', 'Educational special needs', 'Gender'])

# Vérifier les nouvelles colonnes créées
print(encoded_data.columns)

from sklearn.preprocessing import MinMaxScaler

# Normaliser les données en utilisant la normalisation MinMax
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(encoded_data)

# Créer un nouveau dataframe pour les données normalisées
normalized_data = pd.DataFrame(scaled_data, columns=encoded_data.columns)

# Vérifier les données normalisées
print(normalized_data.head())



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Séparer les caractéristiques et les étiquettes
X = data.drop('Target', axis=1)
y = data['Target']

# Diviser les données en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialiser les classifieurs
dtc = DecisionTreeClassifier(random_state=42)
gnb = GaussianNB()
rfc = RandomForestClassifier(random_state=42)

# Entraîner les classifieurs avec les données d'entraînement
dtc.fit(X_train, y_train)
gnb.fit(X_train, y_train)
rfc.fit(X_train, y_train)

# Prédire les étiquettes des données de test
y_pred_dtc = dtc.predict(X_test)
y_pred_gnb = gnb.predict(X_test)
y_pred_rfc = rfc.predict(X_test)

# Calculer l'exactitude des classifieurs
accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)

# Afficher l'exactitude des classifieurs
print("Decision Tree Classifier Accuracy:", accuracy_dtc)
print("Gaussian Naive Bayes Classifier Accuracy:", accuracy_gnb)
print("Random Forest Classifier Accuracy:", accuracy_rfc)

# Sélectionner le classifieur le plus efficace
best_classifier = max(accuracy_dtc, accuracy_gnb, accuracy_rfc)
if best_classifier == accuracy_dtc:
    print("Decision Tree Classifier is the best")
elif best_classifier == accuracy_gnb:
    print("Gaussian Naive Bayes Classifier is the best")
else:
    print("Random Forest Classifier is the best")



import streamlit as st
import pandas as pd
import numpy as np

# Create a Streamlit app to get input data and display predictions
st.title("Predicting Student Retention")

# Get input data from user
age = st.number_input("Age at enrollment:", min_value=16, max_value=60, value=18, step=1)
gender = st.selectbox("Gender:", options=["Male", "Female"])
income = st.number_input("Income:", min_value=0, max_value=200000, value=50000, step=1000)
course = st.selectbox("Course:", options=["Science", "Arts", "Commerce"])
units = st.number_input("Curricular units 1st sem (approved):", min_value=0, max_value=10, value=5, step=1)
application_mode = st.selectbox("Application mode:", options=["Online", "Offline"])
application_order = st.selectbox("Application order:", options=["First", "Second", "Third"])

# Create a Pandas DataFrame with the input data
input_data = pd.DataFrame({
    "Age at enrollment": [age],
    "Gender": [gender],
    "Income": [income],
    "Course": [course],
    "Curricular units 1st sem (approved)": [units],
    "Application mode": [application_mode],
    "Application order": [application_order]
})

# Define a function to predict the retention status
def predict_retention(input_data):
    if input_data["Gender"].values[0] == "Male":
        if input_data["Income"].values[0] < 30000:
            return "Not retained"
        else:
            return "Retained"
    else:
        if input_data["Course"].values[0] == "Commerce":
            if input_data["Curricular units 1st sem (approved)"].values[0] < 4:
                return "Not retained"
            else:
                return "Retained"
        else:
            if input_data["Age at enrollment"].values[0] < 20:
                return "Retained"
            else:
                return "Not retained"

# Make a prediction on the input data
prediction = predict_retention(input_data)

# Display the prediction to the user
st.write("Predicted retention status:", prediction)

