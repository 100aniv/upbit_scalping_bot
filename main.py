# 디렉터리: indicators
# 파일: indicators.py

import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    고급 기술적 지표 모듈 (7종 고급화된 지표 포함)
    - RSI, MACD, VWAP, Bollinger Bands, ATR, EMA, Fibonacci Retracement
    - NumPy 기반 벡터 연산 최적화
    - AI 학습에 최적화
    """

    def calculate_rsi(self, data, period=14):
        """상대강도지수 (RSI) 계산 - NumPy 최적화"""
        delta = data['close'].values
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        data['rsi'] = np.concatenate((np.full(period - 1, np.nan), rsi))
        return data

    def calculate_macd(self, data, short_period=12, long_period=26, signal_period=9):
        """MACD 계산 (고급화 및 NumPy 최적화)"""
        short_ema = data['close'].ewm(span=short_period, adjust=False).mean()
        long_ema = data['close'].ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
        data['macd'] = macd
        data['macd_signal'] = macd_signal
        return data

    def calculate_vwap(self, data):
        """VWAP 계산 (벡터화 및 메모리 최적화)"""
        cumulative_volume = np.cumsum(data['volume'].values)
        cumulative_price_volume = np.cumsum(data['close'].values * data['volume'].values)
        data['vwap'] = cumulative_price_volume / (cumulative_volume + 1e-10)
        return data

    def calculate_bollinger_bands(self, data, period=20):
        """볼린저 밴드 계산 (NumPy 벡터화)"""
        rolling_mean = data['close'].rolling(window=period).mean().values
        rolling_std = data['close'].rolling(window=period).std().values
        data['bb_middle'] = rolling_mean
        data['bb_upper'] = rolling_mean + (2 * rolling_std)
        data['bb_lower'] = rolling_mean - (2 * rolling_std)
        return data

    def calculate_atr(self, data, period=14):
        """ATR 계산 (NumPy 벡터화 및 최적화)"""
        high_low = data['high'].values - data['low'].values
        high_close = np.abs(data['high'].values - data['close'].shift().values)
        low_close = np.abs(data['low'].values - data['close'].shift().values)
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = np.convolve(tr, np.ones(period) / period, mode='valid')
        data['atr'] = np.concatenate((np.full(period - 1, np.nan), atr))
        return data

    def calculate_ema(self, data, period=14):
        """EMA 계산 (벡터화 및 고속 처리)"""
        data['ema'] = data['close'].ewm(span=period, adjust=False).mean()
        return data

    def calculate_fibonacci(self, data):
        """피보나치 되돌림 계산 (벡터화 적용)"""
        high = data['high'].max()
        low = data['low'].min()
        diff = high - low
        levels = [0.382, 0.5, 0.618]
        for level in levels:
            data[f'fibonacci_{int(level*100)}'] = high - diff * level
        return data

# 고급화 테스트 예제
if __name__ == "__main__":
    data = pd.DataFrame({
        'close': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000
    })
    indicators = TechnicalIndicators()
    data = indicators.calculate_rsi(data)
    data = indicators.calculate_macd(data)
    data = indicators.calculate_vwap(data)
    data = indicators.calculate_bollinger_bands(data)
    data = indicators.calculate_atr(data)
    data = indicators.calculate_ema(data)
    data = indicators.calculate_fibonacci(data)
    print(data.head())
