import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
import urllib3
import os

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['REQUESTS_CA_BUNDLE'] = '/home/yneguey/cert/ca.crt'
os.environ['SSL_CERT_FILE'] = '/home/yneguey/cert/mlflow.crt'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

def train_iris_model():
    """Обучение модели Iris"""
    print("Обучение модели Iris...")
    
    # Загрузка данных
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    mlflow.set_experiment("Iris Classification")
    
    with mlflow.start_run() as run:
        print(f"ID запуска: {run.info.run_id}")
        
        # Параметры модели
        n_estimators = 100
        max_depth = 5
        random_state = 42
        
        # Логирование параметров
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Обучение модели
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Предсказания и метрики
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Логирование метрик
        mlflow.log_metric("accuracy", accuracy)
        
        # Логирование модели - ВАЖНО!
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # Явно указываем путь
            registered_model_name="Iris_Production"  # Сразу регистрируем
        )
        
        print(f"Модель обучена с точностью: {accuracy:.4f}")
        print(f"Модель сохранена в артефактах")
        
        return model, accuracy

if __name__ == "__main__":
    train_iris_model()
