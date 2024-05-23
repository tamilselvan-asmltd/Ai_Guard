
import  subprocess
from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import joblib
import re
import requests
import json

# Define the functions

def install_dependencies():
    subprocess.run(["pip3", "install", "scikit-learn", "joblib", "pandas", "requests"])

def load_credit_card_model():
    model = joblib.load('/opt/airflow/dags/credit_card_model.joblib')
    vectorizer = joblib.load('/opt/airflow/dags/credit_card_vectorizer.joblib')
    return model, vectorizer

def predict_credit_card_from_logs():
    # Load the trained model
    model, vectorizer = load_credit_card_model()
    
    # Read the content of the logs.log file
    with open('/opt/airflow/dags/logs.log', 'r') as file:
        input_string = file.read().strip()  # Read the entire content and strip any leading/trailing whitespace
    
    # Perform prediction
    input_vectorized = vectorizer.transform([input_string])
    prediction = model.predict(input_vectorized)
    
    # Print the prediction
    if prediction[0] == 1:
        result = "Yes"
    else:
        result = "No"

    return result
  

def load_phone_number_model():
    model = joblib.load('/opt/airflow/dags/phone_number_model.joblib')
    return model

def extract_phone_number_features(text):
    if re.match(r'\d{3}-\d{3}-\d{3}', text):
        return 1
    else:
        return 0

def predict_phone_number(**context):
    input_string = context['ti'].xcom_pull(task_ids='read_logs_file')
    model = load_phone_number_model()
    tokens = input_string.split(" ")
    for token in tokens:
        features = extract_phone_number_features(token)
        prediction = model.predict([[features]])
        if prediction == 1:
            result = "yes"
            break
    else:
        result = "No"
    return result

def load_email_model():
    model = joblib.load('/opt/airflow/dags/email_classifier_model.joblib')
    vectorizer = joblib.load('/opt/airflow/dags/vectorizer.joblib')
    return model, vectorizer

def predict_email(**context):
    input_string = context['ti'].xcom_pull(task_ids='read_logs_file')
    model, vectorizer = load_email_model()
    input_text_vectorized = vectorizer.transform([input_string])
    prediction = model.predict(input_text_vectorized)
    if prediction[0] == 1:
        result = "yes"
    else:
        result = "No"
    return result

def post_request(**context):
    phone_number_prediction = context['ti'].xcom_pull(task_ids='predict_phone_number')
    email_prediction = context['ti'].xcom_pull(task_ids='predict_email')
    credit_card_prediction = context['ti'].xcom_pull(task_ids='predict_credit_card')
    
    combined_result = f"Phone Number Prediction: {phone_number_prediction}, Email Prediction: {email_prediction}, Credit Card Prediction: {credit_card_prediction}"

    api_url = "http://192.168.0.7:5000/save_data"
    data = {"data": combined_result}
    response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

    if response.status_code == 200:
        print(f"Data successfully sent to API: {combined_result}")
    else:
        print(f"Failed to send data to API. Status code: {response.status_code}")

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 23),
    'retries': 1,
}

dag = DAG(
    'all_in_one_v1',
    default_args=default_args,
    description='DAG for credit card prediction, phone number prediction, and email classifier prediction',
    schedule_interval=None,
)

# Define the tasks
install_task = PythonOperator(
    task_id='install_dependencies',
    python_callable=install_dependencies,
    dag=dag,
)

read_logs_task = PythonOperator(
    task_id='read_logs_file',
    python_callable=lambda: open('/opt/airflow/dags/logs.log', 'r').read().strip(),
    dag=dag,
)

predict_credit_card_task = PythonOperator(
    task_id='predict_credit_card',
    python_callable=predict_credit_card_from_logs,
    dag=dag,
)

predict_phone_task = PythonOperator(
    task_id='predict_phone_number',
    python_callable=predict_phone_number,
    provide_context=True,
    dag=dag,
)

predict_email_task = PythonOperator(
    task_id='predict_email',
    python_callable=predict_email,
    provide_context=True,
    dag=dag,
)

post_request_task = PythonOperator(
    task_id='post_request',
    python_callable=post_request,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
install_task >> read_logs_task >> [predict_credit_card_task, predict_phone_task, predict_email_task] >> post_request_task
