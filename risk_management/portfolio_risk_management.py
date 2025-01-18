# 디렉터리: risk_management
# 파일: portfolio_risk_management.py

import pandas as pd
import numpy as np

class PortfolioRiskManagement:
    """
    고급 포트폴리오 및 리스크 관리 모듈
    - 포트폴리오 최적화 (Mean-Variance Optimization)
    - Value-at-Risk (VaR) 계산
    - 포트폴리오 다각화 분석
    - 손절매 및 트레일링 스탑 관리
    """

    def __init__(self, data):
        self.data = data

    def calculate_portfolio_weights(self, returns, risk_free_rate=0.01):
        """
        포트폴리오 최적화 (Sharpe Ratio 기반)
        """
        cov_matrix = returns.cov()
        mean_returns = returns.mean()

        inv_cov_matrix = np.linalg.inv(cov_matrix)
        weights = inv_cov_matrix.dot(mean_returns)
        weights /= np.sum(weights)

        sharpe_ratio = (mean_returns.dot(weights) - risk_free_rate) / np.sqrt(weights.dot(cov_matrix).dot(weights))
        print(f"최적화된 Sharpe Ratio: {sharpe_ratio:.2f}")

        return weights

    def calculate_var(self, confidence_level=0.95):
        """
        Value-at-Risk (VaR) 계산
        """
        returns = self.data['close'].pct_change().dropna()
        mean_return = returns.mean()
        std_dev = returns.std()
        var = np.percentile(returns, 100 * (1 - confidence_level))
        print(f"{confidence_level*100}% 수준의 Value-at-Risk: {var:.4f}")
        return var

    def calculate_diversification_ratio(self):
        """
        포트폴리오 다각화 비율 계산
        """
        returns = self.data.pct_change().dropna()
        portfolio_volatility = np.sqrt(returns.cov().sum().sum())
        individual_volatility = np.sum(returns.std())

        diversification_ratio = individual_volatility / portfolio_volatility
        print(f"포트폴리오 다각화 비율: {diversification_ratio:.2f}")
        return diversification_ratio

    def apply_stop_loss(self, threshold=0.05):
        """
        손절매 적용
        """
        self.data['stop_loss_triggered'] = self.data['close'].pct_change() <= -threshold
        return self.data

    def apply_trailing_stop(self, trail_percent=0.03):
        """
        트레일링 스탑 적용
        """
        max_price = self.data['close'].cummax()
        self.data['trailing_stop_triggered'] = (self.data['close'] < max_price * (1 - trail_percent))
        return self.data

# 테스트 및 예제 코드
if __name__ == "__main__":
    data = pd.DataFrame({
        'close': np.random.rand(100) * 100
    })

    risk_manager = PortfolioRiskManagement(data)
    returns = data['close'].pct_change().dropna().to_frame()
    weights = risk_manager.calculate_portfolio_weights(returns)
    var = risk_manager.calculate_var()
    diversification_ratio = risk_manager.calculate_diversification_ratio()
    data = risk_manager.apply_stop_loss()
    data = risk_manager.apply_trailing_stop()
    print("포트폴리오 리스크 관리 테스트 완료!")
