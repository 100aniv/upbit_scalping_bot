import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def load_and_preprocess_data(file_path):
    """
    데이터를 로딩하고 전처리 (스케일링 추가)
    """
    data = pd.read_csv(file_path)
    X = data.drop(columns=['target'])
    y = data['target']

    # 스케일링 적용
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def hyperparameter_tuning(X_train, y_train):
    """
    랜덤 포레스트 하이퍼파라미터 최적화
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_ensemble(X_train, y_train):
    """
    앙상블 기법 적용: Random Forest + Gradient Boosting + Voting Classifier
    """
    rf_best = hyperparameter_tuning(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=100)
    
    ensemble_model = VotingClassifier(estimators=[
        ('rf', rf_best),
        ('gb', gb)
    ], voting='soft')
    
    ensemble_model.fit(X_train, y_train)
    return ensemble_model

def evaluate_model(model, X_test, y_test):
    """
    모델 성능 평가
    """
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

def save_model(model, file_path):
    """모델 저장"""
    joblib.dump(model, file_path)

if __name__ == "__main__":
    file_path = "data/backtesting_data.csv"
    model_save_path = "ai_models/optimized_random_forest.pkl"

    # 데이터 로딩 및 전처리
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # 모델 학습 (하이퍼파라미터 최적화 및 앙상블 적용)
    model = train_ensemble(X_train, y_train)

    # 성능 평가
    evaluate_model(model, X_test, y_test)

    # 모델 저장
    save_model(model, model_save_path)
    print("최적화된 모델 저장 완료!")
