import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset from CSV
df = pd.read_csv('credit_card_dataset.csv')

# Split the dataset into features (text) and labels
X = df['text']
y = df['label']

# Vectorize the text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Save the trained model and vectorizer using joblib
joblib.dump(model, 'credit_card_model.joblib')
joblib.dump(vectorizer, 'credit_card_vectorizer.joblib')

# Function to predict whether the input string is a credit card number
def predict_credit_card(input_string):
    input_vectorized = vectorizer.transform([input_string])
    prediction = model.predict(input_vectorized)
    return prediction[0]

# Vectorize the input text
input_text_vectorized = vectorizer.transform(["iam a tamilse casfio fasdvasv 23523-523-5325-235-53-235-5"])

# Predict using the trained model
prediction = model.predict(input_text_vectorized)

# Print the prediction
if prediction[0] == 1:
    print("The input is predicted to be a credit card number.")
else:
    print("The input is predicted not to be a credit card number.")
