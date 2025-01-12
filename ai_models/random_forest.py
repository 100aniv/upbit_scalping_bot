import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import shap

# 데이터 로딩 및 전처리 함수 정의
# 이 함수는 데이터를 로딩하고 전처리한 후 학습/테스트 세트로 나눕니다.
def load_and_preprocess_data(file_path):
    """
    주어진 CSV 파일에서 데이터를 로딩하고, 전처리 및 학습/테스트 세트로 분할
    Args:
        file_path (str): CSV 파일 경로
    Returns:
        X_train, X_test, y_train, y_test: 학습 및 테스트 데이터
    """
    # CSV 파일 로드
    data = pd.read_csv(file_path)
    # 특징(Features)와 라벨(Target) 분리
    X = data.drop(columns=['target'])
    y = data['target']
    # 데이터를 학습 및 테스트 세트로 나누기 (80% 학습, 20% 테스트)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 랜덤 포레스트 모델 학습 함수 정의
# Random Forest 모델을 초기화하고 학습시킵니다.
def train_random_forest(X_train, y_train):
    """
    랜덤 포레스트 모델을 초기화하고 학습
    Args:
        X_train (pd.DataFrame): 학습 데이터
        y_train (pd.Series): 학습 라벨
    Returns:
        model (RandomForestClassifier): 학습된 모델
    """
    # Random Forest Classifier 초기화 및 학습
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

# 모델 평가 함수
# 정확도 및 분류 보고서를 출력합니다.
def evaluate_model(model, X_test, y_test):
    """
    학습된 모델을 평가하고 성능을 출력
    Args:
        model (RandomForestClassifier): 학습된 모델
        X_test (pd.DataFrame): 테스트 데이터
        y_test (pd.Series): 테스트 라벨
    Returns:
        None
    """
    # 예측 수행
    y_pred = model.predict(X_test)
    # 정확도 및 분류 보고서 출력
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

# 모델 저장 및 불러오기
# joblib을 사용하여 모델을 저장하고 로딩합니다.
def save_model(model, file_path):
    """
    학습된 모델을 지정된 경로에 저장
    Args:
        model (RandomForestClassifier): 학습된 모델
        file_path (str): 모델을 저장할 파일 경로
    Returns:
        None
    """
    joblib.dump(model, file_path)


def load_model(file_path):
    """
    저장된 모델을 로드
    Args:
        file_path (str): 모델 파일 경로
    Returns:
        model (RandomForestClassifier): 로드된 모델
    """
    return joblib.load(file_path)

# SHAP를 이용한 모델 설명
# SHAP 값을 계산하고 상위 특징들을 시각화합니다.
def explain_model(model, X_train):
    """
    SHAP 값을 이용하여 모델 설명
    Args:
        model (RandomForestClassifier): 학습된 모델
        X_train (pd.DataFrame): 학습 데이터
    Returns:
        None
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)

# 메인 실행 흐름
if __name__ == "__main__":
    # 데이터 경로 설정
    file_path = "data/backtesting_data.csv"
    model_save_path = "ai_models/random_forest_model.pkl"

    # 데이터 로딩 및 전처리
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # 모델 학습
    model = train_random_forest(X_train, y_train)

    # 모델 평가
    evaluate_model(model, X_test, y_test)

    # 모델 저장
    save_model(model, model_save_path)

    # SHAP 모델 설명
    explain_model(model, X_train)

    print("모델 학습, 평가, 저장 및 설명 완료!")
