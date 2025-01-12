# indicators/stochastic.py
import pandas as pd
import talib

class StochasticOscillator:
    def __init__(self, k_period=14, d_period=3):
        """
        스토캐스틱 오실레이터 클래스
        :param k_period: %K 기간 (기본값 14)
        :param d_period: %D 기간 (기본값 3)
        """
        self.k_period = k_period
        self.d_period = d_period

    def calculate_stochastic(self, data):
        """
        스토캐스틱 오실레이터 계산 (TA-Lib 활용)
        :param data: OHLCV 데이터프레임 (high, low, close 필요)
        :return: %K, %D 라인이 추가된 데이터프레임
        """
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            raise ValueError("[Error] 데이터프레임에 필요한 컬럼이 없습니다.")
        
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values

        k, d = talib.STOCH(
            high_prices,
            low_prices,
            close_prices,
            fastk_period=self.k_period,
            slowk_period=self.k_period,
            slowd_period=self.d_period
        )

        data['Stochastic_%K'] = k
        data['Stochastic_%D'] = d
        return data

    def get_stochastic_signal(self, data):
        """
        스토캐스틱 오실레이터 신호 계산
        :param data: 스토캐스틱 지표가 포함된 데이터프레임
        :return: 매수/매도/보류 신호
        """
        if 'Stochastic_%K' not in data.columns or 'Stochastic_%D' not in data.columns:
            raise ValueError("[Error] 스토캐스틱 데이터가 계산되지 않았습니다.")
        
        latest_k = data['Stochastic_%K'].iloc[-1]
        latest_d = data['Stochastic_%D'].iloc[-1]

        if latest_k > 80 and latest_k > latest_d:
            return "Sell"
        elif latest_k < 20 and latest_k < latest_d:
            return "Buy"
        else:
            return "Hold"

if __name__ == "__main__":
    import pyupbit
    data = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=200)
    
    # 스토캐스틱 오실레이터 계산 및 신호 확인
    stoch_indicator = StochasticOscillator()
    data_with_stoch = stoch_indicator.calculate_stochastic(data)
    signal = stoch_indicator.get_stochastic_signal(data_with_stoch)
    
    print(f"스토캐스틱 신호: {signal}")
    print(data_with_stoch.tail())
