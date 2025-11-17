import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import os
import warnings
import urllib3

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['REQUESTS_CA_BUNDLE'] = '/home/yneguey/cert/ca.crt'
os.environ['SSL_CERT_FILE'] = '/home/yneguey/cert/mlflow.crt'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')
mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")


def register_best_model():
    """Регистрация лучшей модели в MLflow Registry"""
    client = mlflow.tracking.MlflowClient()
    try:
        # Поиск лучшего запуска по accuracy
        experiment = client.get_experiment_by_name("Iris Classification")
        if not experiment:
            print("Эксперимент 'Iris Classification' не найден")
            return None
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.accuracy DESC"]
        )
        
        if not runs:
            print("Запуски в эксперименте не найдены")
            return None
            
        best_run = runs[0]

        model_uri = f"runs:/{best_run.info.run_id}/random_forest_model"
        accuracy = best_run.data.metrics['accuracy']

        # Регистрация модели
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="Iris_Production"
        )
        return registered_model, accuracy
        
    except Exception as e:
        return None, None

def test_saved_model(version=None):
    """Тестирование сохраненной модели"""
    print("Тестирование сохраненной модели...")
    
    try:
        # Если версия не указана, используем последнюю
        if version is None:
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions("Iris_Production")
            if latest_versions:
                version = latest_versions[0].version
            else:
                print("Зарегистрированные модели не найдены")
                return None
        
        model_uri = f"models:/Iris_Production/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Модель версии {version} успешно загружена")
        
        # Тестовые данные
        iris = load_iris()
        
        # Берем разные классы для теста
        test_indices = [0, 1, 50, 51, 100, 101]  # По 2 из каждого класса
        X_test = iris.data[test_indices]
        y_test = iris.target[test_indices]
        
        # Предсказания
        predictions = model.predict(X_test)
        
        # Метрики тестирования
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Точность теста: {accuracy:.4f}")
        print(f"Количество тестовых образцов: {len(X_test)}")
        print(f"Классы в тестовых данных: {np.unique(y_test)}")
        print(f"Предсказанные классы: {np.unique(predictions)}")
        
        print("\nОтчет классификации:")
        print(classification_report(y_test, predictions, 
                                  target_names=iris.target_names,
                                  labels=[0, 1, 2]))
        
        # Детальные предсказания
        results = []
        for i, (true, pred) in enumerate(zip(y_test, predictions)):
            results.append({
                "образец": i,
                "истинный_класс": iris.target_names[true],
                "предсказанный_класс": iris.target_names[pred],
                "корректно": true == pred
            })
        
        results_df = pd.DataFrame(results)
        print("\nДетальные предсказания:")
        print(results_df)
        
        # Проверка вероятностей
        print("\nВероятности предсказания для первого образца:")
        probabilities = model.predict_proba(X_test[:1])
        for i, prob in enumerate(probabilities[0]):
            print(f"  {iris.target_names[i]}: {prob:.4f}")
            
        return accuracy

        
    except Exception as e:
        print(f"Ошибка при тестировании модели: {e}")
        return None

def main():
    """Основной пайплайн регистрации и тестирования"""
    # 1. Регистрация лучшей модели
    registered_model, accuracy = register_best_model()
    
    if registered_model:
        # 2. Тестирование зарегистрированной модели
        test_accuracy = test_saved_model(registered_model.version)
        
        print("\nРегистрация и тестирование успешно завершены!")
    else:
        print("Пайплайн завершился ошибкой на этапе регистрации")

if __name__ == "__main__":
    main()