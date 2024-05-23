import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
import joblib

# Load the dataset from CSV
df = pd.read_csv('phone_number_dataset.csv')

# Function to extract features indicating whether the input string follows the pattern ###-###-###
def extract_features(text):
    if re.match(r'\d{3}-\d{3}-\d{3}', text):
        return 1
    else:
        return 0

# Extract features from the text data
df['features'] = df['text'].apply(extract_features)

# Split the dataset into features and labels
X = df['features']
y = df['label']

# No need for CountVectorizer as we're using custom features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train.values.reshape(-1, 1), y_train)  # Reshape X_train for compatibility with LogisticRegression

# Predict on the test data
y_pred = model.predict(X_test.values.reshape(-1, 1))  # Reshape X_test for compatibility with LogisticRegression

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Function to predict whether the input string is a phone number
def predict_phone_number(input_string):
    tokens = input_string.split(" ")  # Split the input string by space
    for token in tokens:
        features = extract_features(token)
        prediction = model.predict([[features]])
        if prediction == 1:
            return "Phone Number"
    return "Not a Phone Number"

# Define the input string
input_string = "afasdv 2345136 26346 435-1-253"

# Predict using the trained model
prediction = predict_phone_number(input_string)

# Print the prediction result
print(f"{input_string}: {prediction}")

# Save the trained model to a file
joblib.dump(model, 'phone_number_model.joblib')
