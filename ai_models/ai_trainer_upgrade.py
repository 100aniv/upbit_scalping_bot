import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from imblearn.over_sampling import SMOTE

class AITrainer:
    def __init__(self, lstm_input_shape, rf_hyperparams=None):
        """
        AI Trainer Class
        :param lstm_input_shape: LSTM 모델 입력 데이터 형태
        :param rf_hyperparams: 랜덤 포레스트 하이퍼파라미터 딕셔너리
        """
        # LSTM 모델 정의
        self.lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=lstm_input_shape),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')

        # Random Forest 모델 정의 및 하이퍼파라미터 설정
        self.rf_hyperparams = rf_hyperparams or {
            'n_estimators': 200,
            'max_depth': 10,
            'random_state': 42
        }
        self.rf_model = RandomForestClassifier(**self.rf_hyperparams)

    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        LSTM 모델 학습
        :param X_train: 학습 데이터
        :param y_train: 학습 라벨
        :param X_val: 검증 데이터
        :param y_val: 검증 라벨
        :param epochs: 에포크 수
        :param batch_size: 배치 크기
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stopping, lr_scheduler]
        )

    def train_random_forest(self, X_train, y_train):
        """
        Random Forest 모델 학습
        :param X_train: 학습 데이터
        :param y_train: 학습 라벨
        """
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        self.rf_model.fit(X_resampled, y_resampled)

    def predict_lstm(self, X_test):
        """
        LSTM 모델 예측
        :param X_test: 테스트 데이터
        :return: 예측 결과
        """
        return self.lstm_model.predict(X_test)

    def predict_random_forest(self, X_test):
        """
        Random Forest 모델 예측
        :param X_test: 테스트 데이터
        :return: 예측 결과
        """
        return self.rf_model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        """
        모델 성능 평가
        :param y_true: 실제 값
        :param y_pred: 예측 값
        :return: 성능 메트릭 딕셔너리
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

def preprocess_data(data, sequence_length, is_classification=False):
    """
    데이터 전처리 및 시퀀스 변환
    :param data: 원본 데이터
    :param sequence_length: 시퀀스 길이
    :param is_classification: 분류 데이터 여부
    :return: (X, y, scaler)
    """
    if is_classification:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i + sequence_length])
        y.append(data_scaled[i + sequence_length])
    return np.array(X), np.array(y), scaler

if __name__ == "__main__":
    # 데이터 생성 및 전처리
    data = np.sin(np.linspace(0, 100, 1000))  # 예제 데이터 (sine wave)
    sequence_length = 50
    X, y, scaler = preprocess_data(data, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 학습 및 평가
    ai_trainer = AITrainer(lstm_input_shape=(sequence_length, 1))
    ai_trainer.train_lstm(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)
    lstm_predictions = ai_trainer.predict_lstm(X_test)

    # 랜덤 포레스트 훈련용 데이터 전처리 (2D로 변환)
    X_rf = X.reshape(X.shape[0], -1)
    X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y, test_size=0.2, shuffle=False)

    ai_trainer.train_random_forest(X_rf_train, y_rf_train)
    rf_predictions = ai_trainer.predict_random_forest(X_rf_test)

    # 평가 결과 출력
    rf_metrics = ai_trainer.evaluate(y_rf_test, rf_predictions)
    print("Random Forest Metrics:", rf_metrics)
