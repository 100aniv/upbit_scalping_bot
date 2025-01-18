# 디렉터리: risk_management
# 파일: trailing_stop.py

import pandas as pd
import numpy as np

class TrailingStop:
    """
    고급 트레일링 스탑 모듈
    - 고정 퍼센트 기반 트레일링 스탑
    - ATR 기반 동적 트레일링 스탑
    - AI 기반 트레일링 스탑 적용
    """

    def __init__(self):
        pass

    def fixed_trailing_stop(self, data, trail_percent=0.03):
        """
        고정 퍼센트 트레일링 스탑 적용
        :param data: 데이터프레임
        :param trail_percent: 트레일링 스탑 퍼센트
        :return: 트레일링 스탑 적용된 데이터프레임
        """
        max_price = data['close'].cummax()
        data['trailing_stop_triggered'] = data['close'] < max_price * (1 - trail_percent)
        return data

    def atr_based_trailing_stop(self, data, atr_period=14, multiplier=1.5):
        """
        ATR 기반 동적 트레일링 스탑
        :param data: 데이터프레임
        :param atr_period: ATR 계산 주기
        :param multiplier: ATR 곱하기 배수
        :return: 트레일링 스탑 적용된 데이터프레임
        """
        data['atr'] = data['close'].rolling(window=atr_period).std()
        data['trailing_stop_level'] = data['close'] - (data['atr'] * multiplier)
        data['trailing_stop_triggered'] = data['close'] < data['trailing_stop_level']
        return data

    def ai_based_trailing_stop(self, data, ai_predictions, confidence_levels):
        """
        AI 기반 트레일링 스탑 적용
        :param data: 데이터프레임
        :param ai_predictions: AI 예측 데이터 (수익 기대값)
        :param confidence_levels: AI 예측 신뢰도
        :return: 트레일링 스탑 적용된 데이터프레임
        """
        risk_adjusted_trailing = ai_predictions * confidence_levels
        max_price = data['close'].cummax()
        data['trailing_stop_triggered'] = data['close'] < (max_price * (1 - risk_adjusted_trailing.mean()))
        return data

# 예제 실행
if __name__ == "__main__":
    data = pd.DataFrame({
        'close': np.random.rand(100) * 100
    })

    trailing_stop = TrailingStop()
    data = trailing_stop.fixed_trailing_stop(data)
    print("고정 퍼센트 트레일링 스탑 적용 완료.")

    data = trailing_stop.atr_based_trailing_stop(data)
    print("ATR 기반 트레일링 스탑 적용 완료.")

    data = trailing_stop.ai_based_trailing_stop(data, ai_predictions=np.random.rand(100) - 0.5, confidence_levels=np.random.rand(100))
    print("AI 기반 트레일링 스탑 적용 완료.")
