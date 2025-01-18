# 디렉터리: ai_models
# 파일: random_forest.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from indicators.indicators import TechnicalIndicators
import joblib

# 데이터 로딩 및 전처리

def load_and_preprocess_data(file_path):
    """
    데이터를 로딩하고 기술적 지표를 추가한 후 전처리
    :param file_path: 데이터 파일 경로
    :return: 학습 및 테스트 데이터셋
    """
    data = pd.read_csv(file_path)

    # 기술적 지표 추가
    indicators = TechnicalIndicators()
    data = indicators.calculate_macd(data)
    data['rsi'] = indicators.calculate_rsi(data)
    data['vwap'] = indicators.calculate_vwap(data)
    data['bb_upper'] = indicators.calculate_bollinger_bands(data)['bb_upper']
    data['atr'] = indicators.calculate_atr(data)
    data['ema'] = indicators.calculate_ema(data)
    data['fibonacci'] = indicators.calculate_fibonacci(data)

    data = data.dropna()
    X = data.drop(columns=['target'])
    y = data['target']

    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 하이퍼파라미터 최적화

def hyperparameter_tuning(X_train, y_train):
    """
    랜덤 포레스트 하이퍼파라미터 최적화
    :param X_train: 학습 데이터
    :param y_train: 라벨 데이터
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

# 앙상블 학습

def train_ensemble(X_train, y_train):
    """
    Random Forest + Gradient Boosting + Voting Classifier를 사용하는 앙상블 학습
    :param X_train: 학습 데이터
    :param y_train: 라벨 데이터
    """
    rf_best = hyperparameter_tuning(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=100)
    ensemble_model = VotingClassifier(estimators=[('rf', rf_best), ('gb', gb)], voting='soft')
    ensemble_model.fit(X_train, y_train)
    return ensemble_model

# 모델 평가

def evaluate_model(model, X_test, y_test):
    """
    모델 성능 평가
    :param model: 훈련된 모델
    :param X_test: 테스트 데이터
    :param y_test: 테스트 라벨 데이터
    """
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

# 모델 저장

def save_model(model, file_path):
    """
    모델 저장
    :param model: 훈련된 모델
    :param file_path: 저장 경로
    """
    joblib.dump(model, file_path)

# 메인 실행부
if __name__ == "__main__":
    file_path = "data/backtesting_data.csv"
    model_save_path = "ai_models/optimized_random_forest.pkl"

    # 데이터 로딩 및 전처리
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # 모델 학습
    model = train_ensemble(X_train, y_train)

    # 모델 성능 평가
    evaluate_model(model, X_test, y_test)

    # 모델 저장
    save_model(model, model_save_path)
    print("최적화된 랜덤 포레스트 모델 저장 완료!")
