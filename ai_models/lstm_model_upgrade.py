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

class EnhancedLSTMModel:
    def __init__(self, input_shape, units=128, dropout_rate=0.3, learning_rate=0.001):
        """
        고급 LSTM 모델 (CNN + Residual Connection + Multi-Head Attention + Bidirectional LSTM)
        :param input_shape: 입력 데이터 형태 (timesteps, features)
        :param units: LSTM 및 GRU 셀의 수
        :param dropout_rate: 드롭아웃 비율
        :param learning_rate: 학습률
        """
        inputs = Input(shape=input_shape)

        # 1D CNN with Residual Connection
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

        # Multi-Head Attention Mechanism
        attention_output = MultiHeadAttention(num_heads=4, key_dim=units)(x, x)
        x = Add()([attention_output, x])  # Residual Connection with Attention
        x = Flatten()(x)

        # Output Layer
        outputs = Dense(1, activation='linear')(x)

        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber())

    def train(self, X, y, epochs=100, batch_size=64, validation_split=0.2):
        """
        모델 학습
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size, 
            validation_split=validation_split, callbacks=[early_stopping, reduce_lr]
        )

    def predict(self, X_test):
        """
        데이터 예측
        """
        return self.model.predict(X_test)

    def save_model(self, file_path):
        """
        모델 저장
        """
        self.model.save(file_path)

    def load_model(self, file_path):
        """
        저장된 모델 로드
        """
        self.model = tf.keras.models.load_model(file_path)

def preprocess_data(data, sequence_length=50):
    """
    데이터 전처리 및 시퀀스 변환
    """
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i + sequence_length])
        y.append(data_scaled[i + sequence_length])

    return np.array(X), np.array(y), scaler

# 테스트 및 예제 코드
if __name__ == "__main__":
    # 예제 데이터 생성 및 전처리
    data = np.sin(np.linspace(0, 100, 1000))  # 예제 시계열 데이터
    sequence_length = 50
    X, y, scaler = preprocess_data(data, sequence_length)

    # 학습 데이터 분할
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 모델 생성 및 학습
    model = EnhancedLSTMModel(input_shape=(sequence_length, 1))
    model.train(X_train, y_train, epochs=50, batch_size=64)

    # 예측 및 성능 확인
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    print(f"예측 결과 (복원): {predictions_rescaled[:5]}")

    # 모델 저장
    model.save_model("enhanced_lstm_model.h5")
