# indicators/macd.py
import pandas as pd
import talib

class MACDIndicator:
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        MACD 지표 클래스
        :param fast_period: MACD의 빠른 기간
        :param slow_period: MACD의 느린 기간
        :param signal_period: 신호선 기간
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate_macd(self, data):
        """
        MACD 계산 (TA-Lib 기반)
        :param data: OHLCV 데이터프레임 (close 컬럼 필요)
        :return: MACD, 신호선, 히스토그램이 추가된 데이터프레임
        """
        if 'close' not in data.columns:
            raise ValueError("[Error] 데이터프레임에 'close' 컬럼이 없습니다.")
        
        close_prices = data['close'].values
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices, 
            fastperiod=self.fast_period, 
            slowperiod=self.slow_period, 
            signalperiod=self.signal_period
        )
        
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        data['MACD_Hist'] = macd_hist
        return data

    def get_macd_signal(self, data):
        """
        MACD 매수/매도 신호 계산
        :param data: MACD가 추가된 데이터프레임
        :return: 매수/매도/보류 신호
        """
        if 'MACD' not in data.columns or 'MACD_Signal' not in data.columns:
            raise ValueError("[Error] MACD 데이터가 계산되지 않았습니다.")
        
        latest_macd = data['MACD'].iloc[-1]
        latest_signal = data['MACD_Signal'].iloc[-1]

        if latest_macd > latest_signal:
            return "Buy"
        elif latest_macd < latest_signal:
            return "Sell"
        else:
            return "Hold"

if __name__ == "__main__":
    import pyupbit
    data = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=200)
    
    # MACD 계산 및 신호 확인
    macd_indicator = MACDIndicator()
    data_with_macd = macd_indicator.calculate_macd(data)
    signal = macd_indicator.get_macd_signal(data_with_macd)
    
    print(f"MACD 신호: {signal}")
    print(data_with_macd.tail())
