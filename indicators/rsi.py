# indicators/rsi.py
import pandas as pd
import talib

class RSIIndicator:
    def __init__(self, period=14):
        """
        RSI 지표 클래스
        :param period: RSI 계산 기간 (기본값 14)
        """
        self.period = period

    def calculate_rsi(self, data):
        """
        RSI 계산 메소드 (TA-Lib 활용)
        :param data: OHLCV 데이터프레임 (close 컬럼 필요)
        :return: RSI가 추가된 데이터프레임
        """
        if 'close' not in data.columns:
            raise ValueError("[Error] 'close' 컬럼이 데이터프레임에 없습니다.")
        
        # RSI 계산
        close_prices = data['close'].values
        data[f'RSI_{self.period}'] = talib.RSI(close_prices, timeperiod=self.period)
        return data

    def get_rsi_signal(self, data):
        """
        RSI 과매수/과매도 신호 계산
        :param data: RSI가 추가된 데이터프레임
        :return: 매수/매도 신호 (Buy, Sell, Hold)
        """
        if f'RSI_{self.period}' not in data.columns:
            raise ValueError("[Error] RSI가 계산되지 않았습니다.")
        
        latest_rsi = data[f'RSI_{self.period}'].iloc[-1]

        if latest_rsi > 70:
            return "Sell"
        elif latest_rsi < 30:
            return "Buy"
        else:
            return "Hold"

if __name__ == "__main__":
    import pyupbit
    data = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=200)
    
    # RSI 계산 및 신호 확인
    rsi_indicator = RSIIndicator(period=14)
    data_with_rsi = rsi_indicator.calculate_rsi(data)
    signal = rsi_indicator.get_rsi_signal(data_with_rsi)
    
    print(f"RSI 신호: {signal}")
    print(data_with_rsi.tail())
