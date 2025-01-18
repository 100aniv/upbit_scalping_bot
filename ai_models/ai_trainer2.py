
# 디렉터리: ai_models
# 파일: lstm_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten, Bidirectional, MultiHeadAttention, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from indicators.indicators import TechnicalIndicators

class EnhancedLSTMModel:
    """
    고급 LSTM 모델 (CNN + Residual Connection + Multi-Head Attention + Bidirectional LSTM + GRU)
    - 기술적 지표 포함
    """

    def __init__(self, input_shape, units=128, dropout_rate=0.3, learning_rate=0.001):
        """
        고급 LSTM 모델 초기화
        :param input_shape: 입력 데이터 형태
        :param units: LSTM 및 GRU 유닛 수
        :param dropout_rate: 드롭아웃 비율
        :param learning_rate: 학습률
        """
        inputs = Input(shape=input_shape)

        # 기술적 지표 추가
        indicators = TechnicalIndicators()
        self.indicators = indicators

        # CNN + Residual Connection
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = Add()([x, inputs])  # Residual Connection
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(dropout_rate)(x)

        # Bidirectional LSTM
        x = Bidirectional(LSTM(units, return_sequences=True))(x)
        x = Dropout(dropout_rate)(x)

        # GRU Layer
        x = GRU(units, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)

        # Multi-Head Attention
        attention_output = MultiHeadAttention(num_heads=4, key_dim=units)(x, x)
        x = Add()([attention_output, x])
        x = Flatten()(x)

        # Output Layer
        outputs = Dense(1, activation='linear')(x)

        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber())

    def train(self, X, y, epochs=100, batch_size=64, validation_split=0.2):
        """
        모델 학습 메서드
        :param X: 학습 데이터
        :param y: 라벨 데이터
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, callbacks=[early_stopping, reduce_lr]
        )

    def predict(self, X_test):
        """
        데이터 예측 메서드
        :param X_test: 예측을 위한 입력 데이터
        """
        return self.model.predict(X_test)

    def save_model(self, file_path):
        """
        모델 저장 메서드
        :param file_path: 모델을 저장할 경로
        """
        self.model.save(file_path)

    def load_model(self, file_path):
        """
        저장된 모델 로드 메서드
        :param file_path: 저장된 모델 파일 경로
        """
        self.model = tf.keras.models.load_model(file_path)

    def preprocess_with_indicators(self, data, sequence_length=50):
        """
        기술적 지표를 적용한 데이터 전처리
        :param data: 원본 시계열 데이터
        """
        data = self.indicators.calculate_macd(data)
        data['rsi'] = self.indicators.calculate_rsi(data)
        data['vwap'] = self.indicators.calculate_vwap(data)
        data['bb_upper'] = self.indicators.calculate_bollinger_bands(data)['bb_upper']
        data['atr'] = self.indicators.calculate_atr(data)
        data['ema'] = self.indicators.calculate_ema(data)
        data['fibonacci'] = self.indicators.calculate_fibonacci(data)

        data = data.dropna()
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.values)

        X, y = [], []
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i:i + sequence_length])
            y.append(data_scaled[i + sequence_length])

        return np.array(X), np.array(y), scaler

# 테스트 및 예제 코드
if __name__ == "__main__":
    import pandas as pd
    data = pd.DataFrame({
        'close': np.sin(np.linspace(0, 100, 1000)),
        'high': np.random.rand(1000) * 100,
        'low': np.random.rand(1000) * 100,
        'volume': np.random.rand(1000) * 1000
    })

    model = EnhancedLSTMModel(input_shape=(50, 1))
    X, y, scaler = model.preprocess_with_indicators(data)
    model.train(X, y, epochs=50, batch_size=64)
    predictions = model.predict(X[:5])
    print(f"예측 결과: {predictions}")
    model.save_model("enhanced_lstm_model.h5")
