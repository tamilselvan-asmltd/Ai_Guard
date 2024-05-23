from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import subprocess
import joblib
import re
import requests
import json

# Define the functions
def install_dependencies():
    subprocess.run(["pip3", "install", "scikit-learn", "joblib", "pandas", "requests"])

def load_phone_number_model():
    model = joblib.load('/opt/airflow/dags/phone_number_model.joblib')
    return model

def extract_phone_number_features(text):
    if re.match(r'\d{3}-\d{3}-\d{3}', text):
        return 1
    else:
        return 0

def predict_phone_number(input_string, model):
    tokens = input_string.split(" ")
    for token in tokens:
        features = extract_phone_number_features(token)
        prediction = model.predict([[features]])
        if prediction == 1:
            result = "Phone Number"
            break
    else:
        result = "Not a Phone Number"
    
    # Send the result to the API
    api_url = "http://192.168.0.6:5000/save_data"
    data = {"data": result}
    response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

    if response.status_code == 200:
        print(f"Data successfully sent to API: {result}")
    else:
        print(f"Failed to send data to API. Status code: {response.status_code}")

    return result

def load_email_model():
    model = joblib.load('/opt/airflow/dags/email_classifier_model.joblib')
    vectorizer = joblib.load('/opt/airflow/dags/vectorizer.joblib')
    return model, vectorizer

def predict_email(input_string, model, vectorizer):
    # Vectorize the input text
    input_text_vectorized = vectorizer.transform([input_string])
    
    # Predict using the loaded model
    prediction = model.predict(input_text_vectorized)
    
    # Determine result
    if prediction[0] == 1:
        result = "The input is predicted to contain an email address."
    else:
        result = "The input is predicted not to contain an email address."
    
    # Send the result to the API
    api_url = "http://192.168.0.6:5000/save_data"
    data = {"data": result}
    response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

    if response.status_code == 200:
        print(f"Data successfully sent to API: {result}")
    else:
        print(f"Failed to send data to API. Status code: {response.status_code}")

    return result

def execute_prediction():
    # Read the input string from logs.log
    with open('/opt/airflow/dags/logs.log', 'r') as file:
        input_string = file.read().strip()  # Read the entire content and strip any leading/trailing whitespace
    
    # Load the phone number model
    phone_number_model = load_phone_number_model()
    
    # Make the phone number prediction
    phone_number_prediction = predict_phone_number(input_string, phone_number_model)
    
    # Load the email classifier model
    email_model, vectorizer = load_email_model()
    
    # Make the email prediction
    email_prediction = predict_email(input_string, email_model, vectorizer)
    
    # Print the results
    print(f"Phone Number Prediction: {phone_number_prediction}")
    print(f"Email Prediction: {email_prediction}")

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 23),
    'retries': 1,
}

dag = DAG(
    'combined_prediction_dag',
    default_args=default_args,
    description='DAG for phone number and email classifier prediction',
    schedule_interval=None,
)

# Define the tasks
install_task = PythonOperator(
    task_id='install_dependencies',
    python_callable=install_dependencies,
    dag=dag,
)

execute_task = PythonOperator(
    task_id='execute_prediction',
    python_callable=execute_prediction,
    dag=dag,
)

# Set task dependencies
install_task >> execute_task
