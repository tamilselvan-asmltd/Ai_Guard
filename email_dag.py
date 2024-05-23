import subprocess
subprocess.run(["pip3", "install", "scikit-learn", "joblib", "pandas", "requests"])
from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator


import requests
import json

# Define the functions
def install_dependencies():
    import subprocess
    subprocess.run(["pip3", "install", "scikit-learn", "joblib", "pandas", "requests"])

def load_model():
    import joblib
    model = joblib.load('/opt/airflow/dags/email_classifier_model.joblib')
    vectorizer = joblib.load('/opt/airflow/dags/vectorizer.joblib')
    return model, vectorizer

def predict_phone_number(input_string, model, vectorizer):
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
    api_url = "http://192.168.100.111:5000/save_data"
    data = {"data": result}
    response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

    if response.status_code == 200:
        print(f"Data successfully sent to API: {result}")
    else:
        print(f"Failed to send data to API. Status code: {response.status_code}")

    return result

'''def execute_prediction():
    input_string = "iam a tamilse casfio fasdvasv 23523-523-5325-235-53-235-5"
    model, vectorizer = load_model()
    prediction = predict_phone_number(input_string, model, vectorizer)
    print(f"{input_string}: {prediction}")

'''


def execute_prediction():
    # Read the input string from logs.log
    with open('/opt/airflow/dags/logs.log', 'r') as file:
        input_string = file.read().strip()  # Read the entire content and strip any leading/trailing whitespace
        model, vectorizer = load_model()
        prediction = predict_phone_number(input_string, model, vectorizer)
        print(f"{input_string}: {prediction}")
    






# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 23),
    'retries': 1,
}

dag = DAG(
    'email_classifier_prediction_dag',
    default_args=default_args,
    description='DAG for email classifier prediction',
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
