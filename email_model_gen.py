import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset from CSV
df = pd.read_csv('email_dataset.csv')

# Split the dataset into features (text) and labels
X = df['text']
y = df['label']

# Vectorize the text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Save the model and the vectorizer using joblib
joblib.dump(model, 'email_classifier_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

# To demonstrate loading and predicting with the saved model and vectorizer
# Load the model and the vectorizer
loaded_model = joblib.load('email_classifier_model.joblib')
loaded_vectorizer = joblib.load('vectorizer.joblib')

# Vectorize the input text using the loaded vectorizer
input_text = "iam a tamilse casfio fasdvasv 23523-523-5325-235-53-235-5"
input_text_vectorized = loaded_vectorizer.transform([input_text])

# Predict using the loaded model
prediction = loaded_model.predict(input_text_vectorized)

# Print the prediction
if prediction[0] == 1:
    print("The input is predicted to contain an email address.")
else:
    print("The input is predicted not to contain an email address.")
