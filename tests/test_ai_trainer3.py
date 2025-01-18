# 디렉터리: tests
# 파일: test_ai_trainer.py

import unittest
import pandas as pd
import numpy as np
from ai_models.ai_trainer import AITrainer
from orders.order_executor import OrderExecutor
from strategies.backtesting import Backtesting
from indicators.indicators import TechnicalIndicators
from risk_management.position_sizing import PositionSizing
from risk_management.stop_loss import StopLoss

class TestAITradingSystem(unittest.TestCase):
    """
    AI 트레이딩 시스템의 전체 테스트
    - 데이터 전처리, 학습, 예측, 주문 실행, 백테스팅 포함
    """

    @classmethod
    def setUpClass(cls):
        cls.trainer = AITrainer()
        cls.data = pd.DataFrame({
            'close': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'volume': np.random.rand(100)
        })
        cls.X, cls.y, _, _ = cls.trainer.preprocess_data(cls.data)
        cls.order_executor = OrderExecutor()
        cls.backtester = Backtesting()
        cls.indicators = TechnicalIndicators()
        cls.position_sizer = PositionSizing()
        cls.stop_loss = StopLoss()

    def test_data_preprocessing(self):
        """데이터 전처리 테스트"""
        try:
            data = self.indicators.calculate_rsi(self.data)
            data = self.indicators.calculate_macd(data)
            self.assertTrue('rsi' in data.columns and 'macd' in data.columns)
        except Exception as e:
            self.fail(f"데이터 전처리 오류 발생: {e}")

    def test_lstm_training(self):
        """LSTM 모델 학습 테스트"""
        try:
            self.trainer.train_lstm(self.X, self.y, epochs=1, batch_size=16)
        except Exception as e:
            self.fail(f"LSTM 학습 오류 발생: {e}")

    def test_random_forest_training(self):
        """Random Forest 학습 테스트"""
        try:
            self.trainer.train_random_forest(self.X, self.y)
        except Exception as e:
            self.fail(f"Random Forest 학습 오류 발생: {e}")

    def test_order_execution(self):
        """주문 실행 테스트"""
        try:
            order_result = self.order_executor.execute_trades(self.data)
            self.assertTrue(order_result)
        except Exception as e:
            self.fail(f"주문 실행 오류 발생: {e}")

    def test_position_sizing(self):
        """포지션 크기 조절 테스트"""
        try:
            optimized_size = self.position_sizer.optimize_position_sizing(self.data)
            self.assertTrue(optimized_size > 0)
        except Exception as e:
            self.fail(f"포지션 크기 조절 오류 발생: {e}")

    def test_stop_loss(self):
        """손절매 테스트"""
        try:
            stop_loss_triggered = self.stop_loss.apply_stop_loss(self.data)
            self.assertIn('stop_loss_triggered', self.data.columns)
        except Exception as e:
            self.fail(f"손절매 적용 오류 발생: {e}")

    def test_backtesting(self):
        """백테스팅 테스트"""
        try:
            backtest_result = self.backtester.run_parallel_backtest(self.data)
            self.assertIsNotNone(backtest_result)
        except Exception as e:
            self.fail(f"백테스팅 오류 발생: {e}")

if __name__ == "__main__":
    unittest.main()
