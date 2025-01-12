import unittest
import numpy as np
from ai_models.ai_trainer import AITrainer

class TestAITrainer(unittest.TestCase):

    def setUp(self):
        """테스트를 위한 인스턴스 및 데이터 준비"""
        self.trainer = AITrainer()
        self.data = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)

    def test_data_preprocessing(self):
        """데이터 전처리 테스트"""
        X, y, scaler = self.trainer.preprocess_data(self.data)
        self.assertEqual(X.shape[1], 50)
        self.assertEqual(y.shape[0], X.shape[0])

    def test_lstm_training(self):
        """LSTM 학습 테스트"""
        X, y, _ = self.trainer.preprocess_data(self.data)
        self.trainer.train_lstm(X, y)
        self.assertTrue(hasattr(self.trainer.lstm_model, 'model'))

    def test_random_forest_training(self):
        """랜덤 포레스트 학습 테스트"""
        X, y, _ = self.trainer.preprocess_data(self.data)
        self.trainer.train_random_forest(X, y)
        self.assertTrue(hasattr(self.trainer.rf_model, 'model'))

    def test_pipeline_execution(self):
        """전체 파이프라인 실행 테스트"""
        result = self.trainer.train_and_evaluate(self.data)
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()
