import numpy as np
from ai_models.lstm_model import LSTMModel
from ai_models.random_forest import RandomForestModel

class EnsembleModel:
    def __init__(self):
        self.lstm_model = LSTMModel()
        self.rf_model = RandomForestModel()

    def predict(self, data):
        # 각 모델의 예측과 신뢰도
        lstm_prediction, lstm_confidence = self.lstm_model.predict(data)
        rf_prediction, rf_confidence = self.rf_model.predict(data)
        
        # 신뢰도 기반 가중 평균 (앙상블)
        total_confidence = lstm_confidence + rf_confidence
        weighted_prediction = (
            (lstm_prediction * lstm_confidence) + 
            (rf_prediction * rf_confidence)
        ) / total_confidence

        return weighted_prediction, total_confidence / 2

# 예제 사용
if __name__ == "__main__":
    model = EnsembleModel()
    example_data = [1.0, 0.8, 0.5]  # 예제 데이터
    prediction, confidence = model.predict(example_data)
    print(f"Ensemble Prediction: {prediction}, Confidence: {confidence}%")
