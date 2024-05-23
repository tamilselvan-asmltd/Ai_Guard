
from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import joblib
import re

# Function to load the trained model and perform prediction
def predict_credit_card_from_logs():
    # Load the trained model
    model = joblib.load('/opt/airflow/dags/credit_card_model.joblib')
    vectorizer = joblib.load('/opt/airflow/dags/credit_card_vectorizer.joblib')
    
    # Read the content of the logs.log file
    with open('/opt/airflow/dags/logs.log', 'r') as file:
        input_string = file.read().strip()  # Read the entire content and strip any leading/trailing whitespace
    
    # Perform prediction
    input_vectorized = vectorizer.transform([input_string])
    prediction = model.predict(input_vectorized)
    
    # Print the prediction
    if prediction[0] == 1:
        print("The input is predicted to be a credit card number.")
    else:
        print("The input is predicted not to be a credit card number.")

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 23),
    'retries': 1,
}

dag = DAG(
    'credit_card_prediction_dag',
    default_args=default_args,
    description='DAG for credit card prediction from logs.log file',
    schedule_interval=None,  # You can specify the schedule interval here
)

# Define the tasks
load_and_predict_task = PythonOperator(
    task_id='load_and_predict_credit_card',
    python_callable=predict_credit_card_from_logs,
    dag=dag,
)

# Set task dependencies
load_and_predict_task
