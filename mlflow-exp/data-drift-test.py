import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfRows, TestNumberOfColumns, TestColumnDrift
import os
import warnings
import urllib3

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['REQUESTS_CA_BUNDLE'] = '/home/yneguey/cert/ca.crt'
os.environ['SSL_CERT_FILE'] = '/home/yneguey/cert/mlflow.crt'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

def create_data_drift_report():
    # Загрузка и подготовка данных
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Разделение на reference и current данные
    reference_data = df.sample(frac=0.7, random_state=42)
    current_data = df.drop(reference_data.index)
    
    # Создание отчета о data drift
    data_drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    # Сохранение отчета
    data_drift_report.save_html("data_drift_report.html")
    
    # Создание тестов
    data_drift_test_suite = TestSuite(tests=[
        TestNumberOfRows(),
        TestNumberOfColumns(),
        TestColumnDrift(column_name='sepal length (cm)'),
        TestColumnDrift(column_name='sepal width (cm)'),
        TestColumnDrift(column_name='petal length (cm)'),
        TestColumnDrift(column_name='petal width (cm)')
    ])
    
    data_drift_test_suite.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    data_drift_test_suite.save_html("data_drift_tests.html")
    
    return data_drift_report, data_drift_test_suite

def log_evidently_to_mlflow():
    mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")
    mlflow.set_experiment("Data Quality Monitoring")
    
    with mlflow.start_run():
        report, test_suite = create_data_drift_report()
        
        # Логирование результатов в MLflow
        mlflow.log_artifact("data_drift_report.html")
        mlflow.log_artifact("data_drift_tests.html")
        
        # Извлечение и логирование метрик drift
        report_json = report.json()
        mlflow.log_text(report_json, "data_drift_report.json")
        
        print("Data drift analysis completed and logged to MLflow")

if __name__ == "__main__":
    log_evidently_to_mlflow()