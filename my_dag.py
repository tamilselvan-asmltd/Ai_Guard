
from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import joblib
import re
import requests
import json

# Define the functions
def install_dependencies():
    import subprocess
    subprocess.run(["pip3", "install", "scikit-learn", "joblib", "pandas", "requests"])

def load_model():
    model = joblib.load('/opt/airflow/dags/phone_number_model.joblib')
    return model

def extract_features(text):
    if re.match(r'\d{3}-\d{3}-\d{3}', text):
        return 1
    else:
        return 0

def predict_phone_number(input_string, model):
    tokens = input_string.split(" ")
    for token in tokens:
        features = extract_features(token)
        prediction = model.predict([[features]])
        if prediction == 1:
            result = "Phone Number"
            break
    else:
        result = "Not a Phone Number"

    # Send the result to the API
    api_url = "http://192.168.100.111:5000/save_data"
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
    
    # Load the model
    model = load_model()
    
    # Make the prediction
    prediction = predict_phone_number(input_string, model)
    
    # Print the result
    print(f"{input_string}: {prediction}")

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 23),
    'retries': 1,
}

dag = DAG(
    'phone_number_prediction_dag',
    default_args=default_args,
    description='DAG for phone number prediction',
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
