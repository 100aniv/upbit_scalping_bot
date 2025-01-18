# 디렉터리: tests
# 파일: test_risk_management.py

import unittest
import pandas as pd
import numpy as np
from risk_management.position_sizing import PositionSizing
from risk_management.stop_loss import StopLoss
from risk_management.trailing_stop import TrailingStop
from risk_management.portfolio_risk_management import PortfolioRiskManagement

class TestRiskManagement(unittest.TestCase):
    """
    리스크 관리 모듈 통합 테스트
    - 포지션 크기, 손절매, 트레일링 스탑, 포트폴리오 최적화 포함
    """

    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame({
            'close': np.random.rand(100) * 100
        })
        cls.position_sizer = PositionSizing()
        cls.stop_loss = StopLoss()
        cls.trailing_stop = TrailingStop()
        cls.risk_manager = PortfolioRiskManagement(cls.data)

    def test_position_sizing(self):
        """포지션 크기 테스트 (고급화)"""
        size = self.position_sizer.volatility_adjusted_sizing(self.data, capital=10000)
        self.assertGreater(size, 0)

    def test_stop_loss(self):
        """손절매 테스트 (고급화)"""
        updated_data = self.stop_loss.fixed_stop_loss(self.data)
        self.assertIn('stop_loss_triggered', updated_data.columns)

    def test_trailing_stop(self):
        """트레일링 스탑 테스트 (고급화)"""
        updated_data = self.trailing_stop.fixed_trailing_stop(self.data)
        self.assertIn('trailing_stop_triggered', updated_data.columns)

    def test_portfolio_optimization(self):
        """포트폴리오 최적화 테스트 (고급화)"""
        returns = self.data['close'].pct_change().dropna().to_frame()
        weights = self.risk_manager.calculate_portfolio_weights(returns)
        self.assertAlmostEqual(sum(weights), 1, places=2)

if __name__ == "__main__":
    unittest.main()
