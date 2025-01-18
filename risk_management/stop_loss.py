# 디렉터리: risk_management
# 파일: stop_loss.py

import pandas as pd
import numpy as np

class StopLoss:
    """
    고급 손절매 모듈
    - 고정 손절매 비율
    - ATR 기반 동적 손절매
    - AI 기반 손절매 적용
    """

    def __init__(self):
        pass

    def fixed_stop_loss(self, data, stop_loss_percent=0.05):
        """
        고정 손절매 비율 적용
        :param data: 데이터프레임
        :param stop_loss_percent: 손절매 비율
        :return: 손절매 적용된 데이터프레임
        """
        data['stop_loss_triggered'] = data['close'].pct_change() <= -stop_loss_percent
        return data

    def atr_based_stop_loss(self, data, atr_period=14, multiplier=1.5):
        """
        ATR 기반 동적 손절매
        :param data: 데이터프레임
        :param atr_period: ATR 계산 주기
        :param multiplier: ATR에 곱할 배수
        :return: 손절매 적용된 데이터프레임
        """
        data['atr'] = data['close'].rolling(window=atr_period).std()
        stop_loss_level = data['close'] - (data['atr'] * multiplier)
        data['stop_loss_triggered'] = data['close'] <= stop_loss_level
        return data

    def ai_based_stop_loss(self, data, ai_predictions, risk_tolerance=0.02):
        """
        AI 예측 기반 손절매
        :param data: 데이터프레임
        :param ai_predictions: AI 예측 데이터 (수익률 예상)
        :param risk_tolerance: 리스크 허용 수준
        :return: 손절매 적용된 데이터프레임
        """
        risk_adjusted_predictions = ai_predictions * (1 - risk_tolerance)
        data['stop_loss_triggered'] = risk_adjusted_predictions < 0
        return data

# 예제 실행
if __name__ == "__main__":
    data = pd.DataFrame({
        'close': np.random.rand(100) * 100
    })

    stop_loss = StopLoss()
    data = stop_loss.fixed_stop_loss(data)
    print("고정 손절매 적용 완료.")

    data = stop_loss.atr_based_stop_loss(data)
    print("ATR 기반 손절매 적용 완료.")

    data = stop_loss.ai_based_stop_loss(data, ai_predictions=np.random.rand(100) - 0.5)
    print("AI 기반 손절매 적용 완료.")
