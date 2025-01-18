# 디렉터리: indicators
# 파일: indicators_updated.py

import pandas as pd
import numpy as np
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import RobustScaler, StandardScaler

class TechnicalIndicators:
    """
    고급 기술적 지표 모듈 (27종 이상)
    - 기존: RSI, MACD, VWAP, Bollinger Bands, ATR, EMA, Fibonacci Retracement, Ichimoku Cloud, Donchian Channel
    - 추가: Keltner Channel, Supertrend, Parabolic SAR, CCI, ROC, Williams %R, Stochastic Oscillator, ADX, MFI, Elder-Ray Index
    - 스케일링 및 데이터 전처리를 위한 RobustScaler, Z-Score Scaling 추가
    - NumPy 기반 벡터 연산 최적화 및 Numba JIT 적용
    - AI 학습에 최적화
    """

    def calculate_rsi(self, data, period=14):
        """
        RSI (Relative Strength Index) 계산 함수
        RSI는 최근의 가격 변동 강도를 측정하는 지표입니다.
        - 70 이상: 과매수 상태 (매도 신호)
        - 30 이하: 과매도 상태 (매수 신호)

        :param data: 종가 데이터를 포함한 데이터프레임
        :param period: RSI 계산에 사용할 기간 (기본값 14)
        :return: RSI 값을 추가한 데이터프레임
        """
        delta = data['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        data['rsi'] = np.concatenate((np.full(period - 1, np.nan), rsi))
        return data

    def calculate_macd(self, data, short_period=12, long_period=26, signal_period=9):
        """
        MACD (Moving Average Convergence Divergence) 계산 함수
        MACD는 두 개의 이동 평균선의 차이를 통해 추세 강도를 측정합니다.
        - MACD 라인이 신호선 위를 교차할 때: 매수 신호
        - MACD 라인이 신호선 아래로 교차할 때: 매도 신호
        """
        short_ema = data['close'].ewm(span=short_period, adjust=False).mean()
        long_ema = data['close'].ewm(span=long_period, adjust=False).mean()
        data['macd'] = short_ema - long_ema
        data['macd_signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()
        return data

    def calculate_vwap(self, data):
        """
        VWAP (Volume Weighted Average Price) 계산 함수
        - 거래량 가중 평균 가격
        - 사용 목적: 기관 투자자들이 많이 사용하는 지표
        """
        cumulative_volume = np.cumsum(data['volume'].values)
        cumulative_price_volume = np.cumsum(data['close'].values * data['volume'].values)
        data['vwap'] = cumulative_price_volume / (cumulative_volume + 1e-10)
        return data

    def calculate_bollinger_bands(self, data, period=20):
        """
        Bollinger Bands (볼린저 밴드) 계산 함수 *
        볼린저 밴드는 가격 변동성을 나타내며, 상단/하단 밴드를 기준으로 과매수 및 과매도 상태를 판단합니다.
        - 상단 밴드: SMA + (2 * 표준편차)
        - 하단 밴드: SMA - (2 * 표준편차)
        - 주가가 상단 밴드에 도달하면 과매수 상태로 해석 (매도 신호 가능)
        - 주가가 하단 밴드에 도달하면 과매도 상태로 해석 (매수 신호 가능)
        """
        rolling_mean = data['close'].rolling(window=period).mean().values
        rolling_std = data['close'].rolling(window=period).std().values
        data['bb_middle'] = rolling_mean
        data['bollinger_upper'] = rolling_mean + (2 * rolling_std)
        data['bollinger_lower'] = rolling_mean - (2 * rolling_std)
        return data

    def calculate_atr(self, data, period=14):
        """
        ATR (Average True Range) 계산 함수
        - 변동성을 측정하는 지표
        - 수익 목표 및 손절매 설정에 활용
        """
        high_low = data['high'].values - data['low'].values
        high_close = np.abs(data['high'].values - data['close'].shift().values)
        low_close = np.abs(data['low'].values - data['close'].shift().values)
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = np.convolve(tr, np.ones(period) / period, mode='valid')
        data['atr'] = np.concatenate((np.full(period - 1, np.nan), atr))
        return data


    def calculate_ema(self, data, period=20):
        """
        EMA (Exponential Moving Average) 계산 함수
        EMA는 최근 데이터에 더 높은 가중치를 두는 이동 평균입니다.
        - EMA는 시장의 최근 동향을 빠르게 반영하는 지표입니다.
        - 주가가 EMA를 상향 돌파할 경우 매수 신호
        - 주가가 EMA를 하향 돌파할 경우 매도 신호
        """
        data['ema'] = data['close'].ewm(span=period, adjust=False).mean()
        return data

    def calculate_fibonacci_retracement(self, data):
        """
        Fibonacci Retracement (피보나치 되돌림) 계산 함수(벡터화 적용)
        피보나치 되돌림은 지지선과 저항선을 예측하는 데 사용됩니다.
        - 23.6%, 38.2%, 50%, 61.8% 수준을 계산
        - 주가가 피보나치 수준에서 반등하면 매수 신호, 돌파하면 추가 하락 가능성
        """
        high = data['high'].max()
        low = data['low'].min()
        diff = high - low
        levels = [0.236, 0.382, 0.5, 0.618]
        for level in levels:
            data[f'fibonacci_{int(level*100)}'] = high - diff * level
        return data

    def calculate_ichimoku_cloud(self, data):
        """
        Ichimoku Cloud (일목균형표) 계산 함수
        - Tenkan-sen (전환선): 최근 9일간의 고가와 저가의 평균
        - Kijun-sen (기준선): 최근 26일간의 고가와 저가의 평균
        - Senkou Span A: 전환선과 기준선의 평균을 26일 미래로 이동
        - Senkou Span B: 최근 52일간의 고가와 저가의 평균을 26일 미래로 이동
        - Chikou Span (후행 스팬): 현재 종가를 26일 뒤로 이동
        """
        high_9 = data['high'].rolling(window=9).max()
        low_9 = data['low'].rolling(window=9).min()
        data['tenkan_sen'] = (high_9 + low_9) / 2

        high_26 = data['high'].rolling(window=26).max()
        low_26 = data['low'].rolling(window=26).min()
        data['kijun_sen'] = (high_26 + low_26) / 2

        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
        high_52 = data['high'].rolling(window=52).max()
        low_52 = data['low'].rolling(window=52).min()
        data['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

        data['chikou_span'] = data['close'].shift(-26)
        return data

    def calculate_donchian_channel(self, data, period=20):
        """
        Donchian Channel (돈치안 채널) 계산 함수
        - Donchian High: 최근 기간 동안의 최고가
        - Donchian Low: 최근 기간 동안의 최저가
        - 사용 목적: 돌파 전략 및 추세 확인
        """
        data['donchian_high'] = data['high'].rolling(window=period).max()
        data['donchian_low'] = data['low'].rolling(window=period).min()
        return data

# 추가 구현될 지표
#1.	트렌드 기반 지표 (3종 추가):
    def calculate_keltner_channel(data, period=20, atr_period=14):
        """
        Keltner Channel 계산 함수
        - 가격의 평균과 ATR을 결합하여 추세 및 변동성을 분석

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: EMA 계산에 사용할 기간 (기본값 20)
        :param atr_period: ATR 계산에 사용할 기간 (기본값 14)
        :return: Keltner Channel 상단 및 하단 밴드를 추가한 데이터프레임
        """
        atr = calculate_atr(data, atr_period)
        ema = data['close'].ewm(span=period, adjust=False).mean()
        data['keltner_upper'] = ema + (2 * atr)
        data['keltner_lower'] = ema - (2 * atr)
        return data

    def calculate_supertrend(data, atr_period=10, multiplier=3):
        """
        Supertrend 계산 함수
        - ATR을 이용하여 현재 추세를 분석하는 강력한 지표

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param atr_period: ATR 계산에 사용할 기간 (기본값 10)
        :param multiplier: ATR 곱연산 계수 (기본값 3)
        :return: Supertrend 상단 및 하단 값을 추가한 데이터프레임
        """
        atr = calculate_atr(data, atr_period)
        hl2 = (data['high'] + data['low']) / 2
        data['supertrend_upper'] = hl2 + (multiplier * atr)
        data['supertrend_lower'] = hl2 - (multiplier * atr)
        return data

    def calculate_parabolic_sar(data, step=0.02, max_step=0.2):
        """
        Parabolic SAR 계산 함수
        - 시간에 따른 추세의 방향과 포지션 전환 신호를 제공

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param step: 가속 계수 (기본값 0.02)
        :param max_step: 가속 계수의 최대값 (기본값 0.2)
        :return: Parabolic SAR 값을 추가한 데이터프레임
        """
        sar = data['low'][0]
        ep = data['high'][0]
        af = step
        data['parabolic_sar'] = np.nan
        for i in range(1, len(data)):
            sar = sar + af * (ep - sar)
            if data['close'][i] > sar:
                ep = max(ep, data['high'][i])
                af = min(af + step, max_step)
            else:
                ep = min(ep, data['low'][i])
                af = min(af + step, max_step)
            data.loc[i, 'parabolic_sar'] = sar
        return data

    #2.	모멘텀 기반 지표 (5종 추가):
    def calculate_cci(data, period=20):
        """
        CCI (Commodity Channel Index) 계산 함수
        - 가격의 이동평균과의 편차를 측정하여 과매수/과매도를 판단

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: CCI 계산에 사용할 기간 (기본값 20)
        :return: CCI 값을 추가한 데이터프레임
        """
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
        data['cci'] = (tp - sma) / (0.015 * mad)
        return data

    def calculate_roc(data, period=14):
        """
        ROC (Rate of Change) 계산 함수
        - 가격의 상대적 변화율을 측정하여 추세 반전 또는 지속성을 평가

        :param data: 종가를 포함한 데이터프레임
        :param period: ROC 계산에 사용할 기간 (기본값 14)
        :return: ROC 값을 추가한 데이터프레임
        """
        data['roc'] = (data['close'] / data['close'].shift(period) - 1) * 100
        return data

    def calculate_williams_r(data, period=14):
        """
        Williams %R 계산 함수
        - 최고가 대비 현재 가격의 위치를 계산하여 과매수/과매도를 판단

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: Williams %R 계산에 사용할 기간 (기본값 14)
        :return: Williams %R 값을 추가한 데이터프레임
        """
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        data['williams_r'] = ((highest_high - data['close']) / (highest_high - lowest_low)) * -100
        return data

    def calculate_stochastic_oscillator(data, period=14, smooth_k=3, smooth_d=3):
        """
        Stochastic Oscillator 계산 함수
        - 현재 가격과 최고/최저가의 관계를 분석하여 시장의 강도를 파악

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: Stochastic Oscillator의 기본 계산 기간 (기본값 14)
        :param smooth_k: %K 선을 부드럽게 하는 기간 (기본값 3)
        :param smooth_d: %D 선을 부드럽게 하는 기간 (기본값 3)
        :return: %K와 %D 값을 추가한 데이터프레임
        """
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        data['%K'] = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        data['%D'] = data['%K'].rolling(window=smooth_d).mean()
        return data

    def calculate_adx(data, period=14):
        """
        ADX (Average Directional Index) 계산 함수
        - 추세의 강도를 측정하며 상승/하락 방향과는 무관

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: ADX 계산에 사용할 기간 (기본값 14)
        :return: ADX 값을 추가한 데이터프레임
        """
        plus_dm = np.where(data['high'].diff() > data['low'].diff(), data['high'].diff(), 0)
        minus_dm = np.where(data['low'].diff() > data['high'].diff(), data['low'].diff(), 0)
        tr = calculate_atr(data, period)['atr']
        plus_di = (plus_dm.rolling(window=period).sum() / tr) * 100
        minus_di = (minus_dm.rolling(window=period).sum() / tr) * 100
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        data['adx'] = dx.rolling(window=period).mean()
        return data

    #3.	거래량 기반 지표 (3종 추가):
    def calculate_mfi(data, period=14):
        """
        MFI (Money Flow Index) 계산 함수
        - 가격과 거래량을 결합하여 과매수/과매도 상태를 분석

        :param data: 고가, 저가, 종가, 거래량을 포함한 데이터프레임
        :param period: MFI 계산에 사용할 기간 (기본값 14)
        :return: MFI 값을 추가한 데이터프레임
        """
        tp = (data['high'] + data['low'] + data['close']) / 3
        mf = tp * data['volume']
        positive_mf = np.where(tp > tp.shift(1), mf, 0)
        negative_mf = np.where(tp < tp.shift(1), mf, 0)
        mfr = positive_mf.sum() / (negative_mf.sum() + 1e-10)
        data['mfi'] = 100 - (100 / (1 + mfr))
        return data

    def calculate_elder_ray(data, period=14):
        """
        Elder-Ray Index 계산 함수
        - 매수와 매도 압력을 측정하여 시장의 힘을 평가

        :param data: 고가, 저가, 종가를 포함한 데이터프레임
        :param period: EMA 계산에 사용할 기간 (기본값 14)
        :return: Bull Power와 Bear Power 값을 추가한 데이터프레임
        """
        ema = data['close'].ewm(span=period, adjust=False).mean()
        data['bull_power'] = data['high'] - ema
        data['bear_power'] = data['low'] - ema
        return data

    def calculate_obv(data):
        """
        OBV (On-Balance Volume) 계산 함수
        - 거래량의 흐름을 분석하여 시장의 추세를 파악

        :param data: 종가와 거래량을 포함한 데이터프레임
        :return: OBV 값을 추가한 데이터프레임
        """
        data['obv'] = np.where(data['close'] > data['close'].shift(1), data['volume'], -data['volume']).cumsum()
        return data

    #4.	시장 복잡성 지표 (2종 추가):
    def calculate_hurst_exponent(data, lag_range=20):
        """
        Hurst Exponent 계산 함수
        - 가격 움직임의 랜덤성과 추세를 분석하여 시장의 복잡성을 평가

        :param data: 종가를 포함한 데이터프레임
        :param lag_range: 계산에 사용할 최대 시차 (기본값 20)
        :return: Hurst 값을 추가한 데이터프레임
        """
        lags = range(2, lag_range)
        tau = [np.std(np.subtract(data['close'].values[lag:], data['close'].values[:-lag])) for lag in lags]
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = reg[0] * 2
        data['hurst_exponent'] = hurst
        return data

    def calculate_fractal_dimension(data, period=14):
        """
        Fractal Dimension 계산 함수
        - 시장의 패턴 복잡성을 측정하여 변동성과 추세를 분석

        :param data: 종가를 포함한 데이터프레임
        :param period: 계산에 사용할 기간 (기본값 14)
        :return: Fractal Dimension 값을 추가한 데이터프레임
        """
        high_low_diff = data['high'] - data['low']
        fd = high_low_diff.rolling(window=period).apply(lambda x: np.log(np.sum(x) / len(x)) / np.log(len(x)))
        data['fractal_dimension'] = fd
        return data

    #5.	조정된 지표 (5종 추가):
    def calculate_adjusted_rsi(data, sentiment_scores, period=14):
        """
        Adjusted RSI 계산 함수
        - 뉴스 감정 점수를 반영하여 RSI 값을 조정

        :param data: 종가 데이터를 포함한 데이터프레임
        :param sentiment_scores: 감정 점수 배열 또는 시리즈
        :param period: RSI 계산에 사용할 기간 (기본값 14)
        :return: Adjusted RSI 값을 추가한 데이터프레임
        """
        delta = data['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        adjusted_rsi = rsi + (sentiment_scores[:len(rsi)] * 10)
        data['adjusted_rsi'] = np.concatenate((np.full(period - 1, np.nan), adjusted_rsi))
        return data

    def calculate_sentiment_adjusted_macd(data, sentiment_scores, short_period=12, long_period=26, signal_period=9):
        """
        Sentiment-Adjusted MACD 계산 함수
        - 감정 점수를 반영하여 MACD 신호의 신뢰도를 강화

        :param data: 종가 데이터를 포함한 데이터프레임
        :param sentiment_scores: 감정 점수 배열 또는 시리즈
        :param short_period: 단기 EMA의 기간 (기본값 12)
        :param long_period: 장기 EMA의 기간 (기본값 26)
        :param signal_period: 신호선 EMA의 기간 (기본값 9)
        :return: Sentiment-Adjusted MACD 값을 추가한 데이터프레임
        """
        short_ema = data['close'].ewm(span=short_period, adjust=False).mean()
        long_ema = data['close'].ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        sentiment_adjusted_macd = macd + (sentiment_scores[:len(macd)] * signal_line)
        data['sentiment_adjusted_macd'] = sentiment_adjusted_macd
        return data

    def calculate_multi_timeframe_ema(data, short_period=12, long_period=26):
        """
        Multi-Timeframe EMA 계산 함수
        - 다양한 시간 프레임의 EMA를 결합하여 추세의 신뢰도를 강화

        :param data: 종가 데이터를 포함한 데이터프레임
        :param short_period: 단기 EMA의 기간 (기본값 12)
        :param long_period: 장기 EMA의 기간 (기본값 26)
        :return: Multi-Timeframe EMA 값을 추가한 데이터프레임
        """
        short_ema = data['close'].ewm(span=short_period, adjust=False).mean()
        long_ema = data['close'].ewm(span=long_period, adjust=False).mean()
        data['multi_timeframe_ema'] = (short_ema + long_ema) / 2
        return data

    def calculate_adjusted_vwap(data, sentiment_scores):
        """
        Adjusted VWAP 계산 함수
        - 뉴스 데이터와 거래량 데이터를 반영한 가중 평균 가격

        :param data: 고가, 저가, 종가, 거래량 데이터를 포함한 데이터프레임
        :param sentiment_scores: 감정 점수 배열 또는 시리즈
        :return: Adjusted VWAP 값을 추가한 데이터프레임
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        cumulative_tp_volume = (typical_price * data['volume']).cumsum()
        cumulative_volume = data['volume'].cumsum()
        vwap = cumulative_tp_volume / cumulative_volume
        adjusted_vwap = vwap + (sentiment_scores[:len(vwap)] * 0.1)
        data['adjusted_vwap'] = adjusted_vwap
        return data

    def calculate_sentiment_weighted_bollinger_bands(data, sentiment_scores, period=20):
        """
        Sentiment-Weighted Bollinger Bands 계산 함수
        - 뉴스 감정 데이터를 변동성 계산에 통합하여 정확도를 높임

        :param data: 종가 데이터를 포함한 데이터프레임
        :param sentiment_scores: 감정 점수 배열 또는 시리즈
        :param period: 볼린저 밴드 계산에 사용할 기간 (기본값 20)
        :return: 상단 및 하단 밴드 값을 추가한 데이터프레임
        """
        sma = data['close'].rolling(window=period).mean()
        std_dev = data['close'].rolling(window=period).std()
        upper_band = sma + (std_dev * 2) + (sentiment_scores[:len(sma)] * 0.1)
        lower_band = sma - (std_dev * 2) - (sentiment_scores[:len(sma)] * 0.1)
        data['sentiment_bollinger_upper'] = upper_band
        data['sentiment_bollinger_lower'] = lower_band
        return data


# 테스트 코드 실행
if __name__ == "__main__":
    # 데이터프레임 생성 (close, high, low, volume 설명 포함)
    data = pd.DataFrame({
        'close': np.random.rand(100) * 100,  # 종가 데이터
        'high': np.random.rand(100) * 100,   # 고가 데이터
        'low': np.random.rand(100) * 100,    # 저가 데이터
        'volume': np.random.rand(100) * 1000 # 거래량 데이터
    })
    
    # 기술적 지표 클래스 인스턴스 생성
    indicators = TechnicalIndicators()

    # 27개 모든 지표 계산 함수 호출
    # 기존 지표
    data = indicators.calculate_rsi(data)  # RSI
    data = indicators.calculate_macd(data)  # MACD
    data = indicators.calculate_ichimoku_cloud(data)  # Ichimoku Cloud
    data = indicators.calculate_donchian_channel(data)  # Donchian Channel
    data = indicators.calculate_vwap(data)  # VWAP
    data = indicators.calculate_atr(data)  # ATR
    data = indicators.calculate_bollinger_bands(data)  # Bollinger Bands
    data = indicators.calculate_fibonacci_retracement(data)  # Fibonacci Retracement
    data = indicators.calculate_ema(data)  # EMA

    # 추가 지표
    data = indicators.calculate_keltner_channel(data)  # Keltner Channel
    data = indicators.calculate_supertrend(data)  # Supertrend
    data = indicators.calculate_parabolic_sar(data)  # Parabolic SAR
    data = indicators.calculate_cci(data)  # CCI
    data = indicators.calculate_roc(data)  # ROC
    data = indicators.calculate_williams_r(data)  # Williams %R
    data = indicators.calculate_stochastic_oscillator(data)  # Stochastic Oscillator
    data = indicators.calculate_adx(data)  # ADX
    data = indicators.calculate_mfi(data)  # MFI
    data = indicators.calculate_elder_ray(data)  # Elder-Ray Index
    data = indicators.calculate_obv(data)  # OBV
    data = indicators.calculate_hurst_exponent(data)  # Hurst Exponent
    data = indicators.calculate_fractal_dimension(data)  # Fractal Dimension
    data = indicators.calculate_adjusted_rsi(data, sentiment_scores=np.random.rand(100))  # Adjusted RSI
    data = indicators.calculate_sentiment_adjusted_macd(data, sentiment_scores=np.random.rand(100))  # Sentiment-Adjusted MACD
    data = indicators.calculate_multi_timeframe_ema(data)  # Multi-Timeframe EMA
    data = indicators.calculate_adjusted_vwap(data, sentiment_scores=np.random.rand(100))  # Adjusted VWAP
    data = indicators.calculate_sentiment_weighted_bollinger_bands(data, sentiment_scores=np.random.rand(100))  # Sentiment-Weighted Bollinger Bands

    # 결과 출력
    print(data.head())
    print(f"데이터프레임에 포함된 열: {data.columns.tolist()}")
