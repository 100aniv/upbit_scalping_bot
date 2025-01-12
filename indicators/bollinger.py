# indicators/bollinger.py
import pandas as pd
import talib

class BollingerBandsIndicator:
    def __init__(self, period=20):
        """
        볼린저 밴드 지표 클래스
        :param period: 볼린저 밴드 계산 기간 (기본값 20)
        """
        self.period = period

    def calculate_bollinger_bands(self, data):
        """
        볼린저 밴드 계산 (TA-Lib 활용)
        :param data: OHLCV 데이터프레임 (close 컬럼 필요)
        :return: 상단, 중간, 하단 밴드가 추가된 데이터프레임
        """
        if 'close' not in data.columns:
            raise ValueError("[Error] 'close' 컬럼이 데이터프레임에 없습니다.")
        
        close_prices = data['close'].values
        upper_band, middle_band, lower_band = talib.BBANDS(
            close_prices, 
            timeperiod=self.period, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )

        data['BB_Upper'] = upper_band
        data['BB_Middle'] = middle_band
        data['BB_Lower'] = lower_band
        return data

    def get_bollinger_signal(self, data):
        """
        볼린저 밴드 신호 계산
        :param data: 볼린저 밴드가 추가된 데이터프레임
        :return: 매수/매도/보류 신호
        """
        if 'BB_Upper' not in data.columns or 'BB_Lower' not in data.columns:
            raise ValueError("[Error] 볼린저 밴드 데이터가 계산되지 않았습니다.")
        
        latest_close = data['close'].iloc[-1]
        latest_upper = data['BB_Upper'].iloc[-1]
        latest_lower = data['BB_Lower'].iloc[-1]

        if latest_close >= latest_upper:
            return "Sell"
        elif latest_close <= latest_lower:
            return "Buy"
        else:
            return "Hold"

if __name__ == "__main__":
    import pyupbit
    data = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=200)
    
    # 볼린저 밴드 계산 및 신호 확인
    bb_indicator = BollingerBandsIndicator()
    data_with_bb = bb_indicator.calculate_bollinger_bands(data)
    signal = bb_indicator.get_bollinger_signal(data_with_bb)
    
    print(f"볼린저 밴드 신호: {signal}")
    print(data_with_bb.tail())
