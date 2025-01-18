# 디렉터리: ai_models
# 파일: ensemble_model.py

import numpy as np
from ai_models.lstm_model import EnhancedLSTMModel
from ai_models.random_forest import RandomForestModel
from indicators.indicators import TechnicalIndicators

class EnsembleModel:
    """
    고급 앙상블 모델 (LSTM + Random Forest)
    - 기술적 지표 7종 포함
    """

    def __init__(self):
        self.lstm_model = EnhancedLSTMModel(input_shape=(50, 1))
        self.rf_model = RandomForestModel()
        self.indicators = TechnicalIndicators()

    def preprocess_with_indicators(self, data):
        """
        기술적 지표를 포함한 데이터 전처리
        """
        data = self.indicators.calculate_macd(data)
        data['rsi'] = self.indicators.calculate_rsi(data)
        data['vwap'] = self.indicators.calculate_vwap(data)
        data['bb_upper'] = self.indicators.calculate_bollinger_bands(data)['bb_upper']
        data['atr'] = self.indicators.calculate_atr(data)
        data['ema'] = self.indicators.calculate_ema(data)
        data['fibonacci'] = self.indicators.calculate_fibonacci(data)
        return data.dropna()

    def predict(self, data):
        """
        앙상블 예측 수행 (LSTM + Random Forest)
        """
        data = self.preprocess_with_indicators(data)
        lstm_prediction = self.lstm_model.predict(data)
        rf_prediction = self.rf_model.predict(data)

        # 신뢰도 기반 가중 평균 (앙상블)
        weighted_prediction = (lstm_prediction + rf_prediction) / 2
        return weighted_prediction

# 테스트 및 예제
if __name__ == "__main__":
    import pandas as pd
    data = pd.DataFrame({
        'close': np.sin(np.linspace(0, 100, 1000)),
        'high': np.random.rand(1000) * 100,
        'low': np.random.rand(1000) * 100,
        'volume': np.random.rand(1000) * 1000
    })

    model = EnsembleModel()
    prediction = model.predict(data)
    print(f"Ensemble Prediction: {prediction[:5]}")
